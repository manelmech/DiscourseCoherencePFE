import torch
import torch.optim as optim
import time
import random
import pickle
from torch.autograd import Variable
from evaluation import *
import progressbar
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import StepLR
from sklearn.utils import class_weight
from numpy import *
from numpy import mean
from numpy import std
from numpy import argmax
from numpy import tensordot
from numpy.linalg import norm
import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
    
# normalize a vector to have unit norm
def normalize(weights):
	# calculate l1 vector norm
	result = norm(weights, 1)
	# check for a vector of all zeros
	if result == 0.0:
		return weights
	# return normalized vector (unit norm)
	return weights / result

# Pour l'apprentissage et test des modèles sans la validation croisée
def train(params, training_docs, test_docs, data, model):
    if params['model_type'] == 'sent_avg':
        training_data, training_labels, train_ids = data.create_doc_sents(training_docs, 'sentence', params['task'], params['train_data_limit'])
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'sentence', params['task'], params['train_data_limit'])

    elif params['model_type'] == 'par_seq':
        training_data, training_labels, train_ids = data.create_doc_sents(training_docs, 'paragraph', params['task'],
                                                                          params['train_data_limit'])
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'paragraph', params['task'], params['train_data_limit'])

    elif params['model_type']=='sem_rel':
        training_data, training_labels, train_ids = data.create_doc_sents(training_docs, 'paragraph', params['task'],
                                                                          params['train_data_limit'])
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'paragraph', params['task'], params['train_data_limit'])

    elif params['model_type']=='cnn_pos_tag':
        training_data, training_labels, train_ids = data.create_doc_sents(training_docs, 'sentence', params['task'],
                                                                          params['train_data_limit'])
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'sentence', params['task'], params['train_data_limit'])

    if USE_CUDA:
        model.cuda()
    if params['train_data_limit'] != -1:
        training_docs = training_docs[:10]
        test_docs = test_docs[:10]
        
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, weight_decay=params['l2_reg'])
    scheduler = None
    if params['lr_decay'] == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif params['lr_decay'] == 'lambda':
        lambda1 = lambda epoch: 0.95 ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda1])

    if params['task'] == 'class':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif params['task'] == 'score_pred':
        loss_fn = torch.nn.MSELoss()

    timestamp = time.time()
    best_test_acc = 0
    best_weights = None
    best_score = 0
    # Tableau utilisé pour la combinaison entre le niveau sémantique entre les phrases et entre les paragraphes
    w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for epoch in range(params['num_epochs']):
        if params['lr_decay'] == 'lambda' or params['lr_decay'] == 'step':
            scheduler.step()
            print(optimizer.param_groups[0]['lr'])
        print("EPOCH "+str(epoch))
        total_loss = 0
        steps = int(len(training_data) / params['batch_size'])
        indices = list(range(len(training_data)))
        random.shuffle(indices)
        bar = progressbar.ProgressBar()
        model.train()
        for step in bar(range(steps)): # Boucler sur le nombre de batchs 

            batch_ind = indices[(step * params["batch_size"]):((step + 1) * params["batch_size"])]
            sentences, orig_batch_labels = data.get_batch(training_data, training_labels, batch_ind, params['model_type'])
            batch_padded, batch_lengths, original_index = data.pad_to_batch(sentences, data.word_to_idx, params['model_type'])
            model.zero_grad()
            if params['model_type']== 'sem_rel':
                y_pred = []
                coherence_pred_sent, coherence_pred_par = model(batch_padded, batch_lengths, original_index)
                # Regrouper les prédictions du niveau phrases et les prédictions du niveau paragraphes dans un seul tableau
                y_pred.append(coherence_pred_sent)
                y_pred.append(coherence_pred_par)
                y_pred = np.array(y_pred)
                for weights in product(w, repeat=2):
                    # Si les poids sont égaux
                    if len(set(weights)) == 1:
                        continue
                    # hack, normalize weight vector
                    weights = normalize(weights)
                    # calculer les poids à associer pour chaque ensemble de prédictions
                    # somme pondérée à travers les deux modèles
                    summed = tensordot(y_pred, weights, axes=((0),(0)))
                    # argmax à travers les classes
                    result = argmax(summed, axis=1)
                    # Calculer l'exactitude
                    score = accuracy_score(orig_batch_labels, result)
                    # Récuperer le meilleur score d'exactitude à travers la combinaison des poids associée
                    if score > best_score:
                        best_score = score
                        best_weights = weights
                        best_weights = list(best_weights)
                final_pred = model(batch_padded, batch_lengths, original_index, weights=best_weights)
                loss = loss_fn(final_pred, Variable(LongTensor(orig_batch_labels)))
            elif params['model_type']== 'cnn_pos_tag': 
                pred_coherence = model(batch_padded, batch_lengths, original_index)
                loss = loss_fn(pred_coherence, Variable(LongTensor(orig_batch_labels))) 
            else: 
                pred_coherence, avg_deg_train= model(batch_padded, batch_lengths, original_index)
                if params['task'] == 'score_pred':
                    loss = loss_fn(pred_coherence, Variable(FloatTensor(orig_batch_labels)))
                else:
                    loss = loss_fn(pred_coherence, Variable(LongTensor(orig_batch_labels)))
            mean_loss = loss / params["batch_size"]
            mean_loss.backward()
            total_loss += loss.cpu().data.numpy()
            optimizer.step()
        current_time = time.time()
        print("Time %-5.2f min" % ((current_time - timestamp) / 60.0))
        print("Train loss: " + str(total_loss))
        output_name = params['model_name'] + '_epoch' + str(epoch)
        if params['model_type'] == 'sent_avg' or params['model_type'] == 'par_seq' or params['model_type']=='sem_rel' or params['model_type']=='cnn_pos_tag':
            if params['model_type']== 'sem_rel' or params['model_type']== 'cnn_pos_tag':
                test_accuracy, test_loss = eval_docs(model, loss_fn, test_data, test_labels, data, params)                                
            elif params['model_type'] == 'sent_avg' or params['model_type'] == 'par_seq':
                test_accuracy, test_loss, global_eval_pred, global_avg_deg_test = eval_docs(model, loss_fn, test_data, test_labels, data, params) 

            print("Test loss: %0.3f" % test_loss)
            if params['task'] == 'score_pred':
                print("Test correlation: %0.5f" % (test_accuracy))
            else:
                print("Test accuracy: %0.2f%%" % (test_accuracy * 100))
        # Récuperer la meilleure exactitude à travers les epochs
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            # sauvegarder le meilleur modèle
            torch.save(model.state_dict(), params['model_dir'] + '/' + params['model_name'] + '_best')
            print('saved model ' + params['model_dir'] + '/' + params['model_name'] + '_best')
            if params['model_type'] == 'sent_avg':
                pickle.dump(model, open('sent_avg.pkl', 'wb'))
            elif params['model_type'] == 'par_seq':
                pickle.dump(model, open('par_seq.pkl', 'wb'))
            elif params['model_type'] == 'cnn_pos_tag':
                pickle.dump(model, open('cnn_pos_tag.pkl', 'wb'))
            elif params['model_type'] == 'sem_rel':
                pickle.dump(model, open('sem_rel.pkl', 'wb'))

        print()
        print("==================== BEST TEST ACCURACY =================================")
        print(best_test_acc)
    return best_test_acc

# Pour l'apprentissage et test des modèles avec la validation croisée
def train_cv(params, data_docs, data, model):
    
    if USE_CUDA:
        model.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, weight_decay=params['l2_reg'])
    scheduler = None
    if params['lr_decay'] == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif params['lr_decay'] == 'lambda':
        lambda1 = lambda epoch: 0.95 ** epoch
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda1])

    if params['task'] == 'class':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif params['task'] == 'score_pred':
        loss_fn = torch.nn.MSELoss()

    timestamp = time.time()
    best_test_acc = 0
    best_weights = None
    best_score = 0
    w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    kfold = StratifiedKFold(n_splits = 5, shuffle = False) 
    for epoch in range(params['num_epochs']):
        fold = 0
        if params['lr_decay'] == 'lambda' or params['lr_decay'] == 'step':
            scheduler.step()
            print(optimizer.param_groups[0]['lr'])
        print("EPOCH " + str(epoch))
        total_loss = 0
        model.train()
        labels = []
        for i in range(len(data_docs)):
            labels.append(data_docs[i].label)
        for train, test in kfold.split(np.zeros(4800), labels):
            training_data = np.array(data_docs)[train]
            test_data = np.array(data_docs)[test]
            training_data, training_labels, train_ids = data.create_doc_sents(training_data, 'paragraph', params['task'],
                                                                          params['train_data_limit'])
            test_data, test_labels, test_ids = data.create_doc_sents(test_data, 'paragraph', params['task'], params['train_data_limit'])

            steps = int(len(training_data) / params['batch_size'])
            indices = list(range(len(training_data)))
            random.shuffle(indices)
            bar = progressbar.ProgressBar()
            for step in bar(range(steps)):

                batch_ind = indices[(step * params["batch_size"]):((step + 1) * params["batch_size"])]
                sentences, orig_batch_labels = data.get_batch(training_data, training_labels, batch_ind, params['model_type'])
                batch_padded, batch_lengths, original_index = data.pad_to_batch(sentences, data.word_to_idx, params['model_type'])
                model.zero_grad()
                if params['model_type']== 'sem_rel':
                    y_pred = []
                    coherence_pred_sent, coherence_pred_par = model(batch_padded, batch_lengths, original_index)
                    # Regrouper les prédictions du niveau phrases et les prédictions du niveau paragraphes dans un seul tableau
                    y_pred.append(coherence_pred_sent)
                    y_pred.append(coherence_pred_par)
                    y_pred = np.array(y_pred)
                    for weights in product(w, repeat=2):
                        # Si les poids sont égaux
                        if len(set(weights)) == 1:
                            continue
                        # hack, normalize weight vector
                        weights = normalize(weights)
                        # calculer les poids à associer pour chaque ensemble de prédictions
                        # somme pondérée à travers les deux modèles
                        summed = tensordot(y_pred, weights, axes=((0),(0)))
                        # argmax à travers les classes
                        result = argmax(summed, axis=1)
                        # Calculer l'exactitude
                        score = accuracy_score(orig_batch_labels, result)
                        # Récuperer le meilleur score d'exactitude à travers la combinaison des poids associée
                        if score > best_score:
                            best_score = score
                            best_weights = weights
                            best_weights = list(best_weights)
                    final_pred = model(batch_padded, batch_lengths, original_index, weights=best_weights)
                    loss_fn = torch.nn.CrossEntropyLoss()
                    loss = loss_fn(final_pred, Variable(LongTensor(orig_batch_labels)))
                    
                else: 
                    pred_coherence = model(batch_padded, batch_lengths, original_index)
                    if params['task'] == 'score_pred':
                        loss = loss_fn(pred_coherence, Variable(FloatTensor(orig_batch_labels)))
                    else:
                        loss = loss_fn(pred_coherence, Variable(LongTensor(orig_batch_labels)))
                mean_loss = loss / params["batch_size"]
                mean_loss.backward()
                total_loss += loss.cpu().data.numpy()
                optimizer.step()
            current_time = time.time()
            print("Time %-5.2f min" % ((current_time - timestamp) / 60.0))
            print("Fold" + str(fold) + " - Train loss: " +str(total_loss))
            if params['model_type'] == 'sent_avg' or params['model_type'] == 'par_seq' or params['model_type']=='sem_rel' or params['model_type']=='cnn_pos_tag':
                
                if params['model_type']== 'sem_rel' or params['model_type'] == 'cnn_pos_tag':
                    test_accuracy, test_loss = eval_docs(model, loss_fn, test_data, test_labels, data, params)                                
                else:
                    test_accuracy, test_loss, global_eval_pred, global_avg_deg_test = eval_docs(model, loss_fn, test_data, test_labels, data, params)                
                print("Fold" + str(fold) +" - Test loss: %0.3f" % test_loss)
                if params['task'] == 'score_pred':
                    print("Test correlation: %0.5f" % (test_accuracy))
                else:
                    print("Fold" + str(fold) +" - Test accuracy: %0.2f%%" % (test_accuracy * 100))
            # Récuperer la meilleure exactitude à travers les epochs
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                # sauvegarder le meilleur modèle
                torch.save(model.state_dict(), params['model_dir'] + '/' + params['model_name'] + '_best')
                print('saved model ' + params['model_dir'] + '/' + params['model_name'] + '_best')
                if params['model_type'] == 'sem_rel' :
                    pickle.dump(model, open('sem_rel_cv.pkl', 'wb'))
                elif params['model_type'] == 'cnn_pos_tag':
                    pickle.dump(model, open('cnn_pos_tag_cv.pkl', 'wb'))
                elif params['model_type'] == 'sent_avg':
                    pickle.dump(model, open('sent_avg_cv.pkl', 'wb'))
                elif params['model_type'] == 'par_seq':
                    pickle.dump(model, open('par_seq_cv.pkl', 'wb'))
            fold += 1
            print()
    print("==================== BEST TEST ACCURACY =================================")
    print(best_test_acc)
    return best_test_acc


def train_fusion(params, data_docs_cnn, data_docs_sem, data, model_cnn, model_sem):
    
    if USE_CUDA:
        model_cnn.cuda()
        model_sem.cuda()
        
    parameters_cnn = filter(lambda p: p.requires_grad, model_cnn.parameters())
    parameters_sem = filter(lambda p: p.requires_grad, model_sem.parameters())
    
    optimizer_cnn = optim.Adam(parameters_cnn, weight_decay=params['l2_reg'])
    optimizer_sem = optim.Adam(parameters_sem, weight_decay=params['l2_reg'])
    
    scheduler_cnn = None
    scheduler_sem = None
    if params['lr_decay'] == 'step':
        scheduler_cnn = StepLR(optimizer_cnn, step_size=30, gamma=0.1)
        scheduler_sem = StepLR(optimizer_sem, step_size=30, gamma=0.1)
    elif params['lr_decay'] == 'lambda':
        lambda1 = lambda epoch: 0.95 ** epoch
        scheduler_cnn = LambdaLR(optimizer_cnn, lr_lambda=[lambda1])
        scheduler_sem = LambdaLR(optimizer_sem, lr_lambda=[lambda1])
    if params['task'] == 'class':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif params['task'] == 'score_pred':
        loss_fn = torch.nn.MSELoss()
    timestamp = time.time()
    best_test_acc = 0
    best_weights = None
    best_score = 0
    w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    kfold = StratifiedKFold(n_splits = 5, shuffle = False) 
    for epoch in range(params['num_epochs']):
        if params['lr_decay'] == 'lambda' or params['lr_decay'] == 'step':
            scheduler_cnn.step()
            scheduler_sem.step()
            print("optimizer CNN_POS_TAG")
            print(optimizer_cnn.param_groups[0]['lr'])
            print("optimizer SEM_REL")
            print(optimizer_sem.param_groups[0]['lr'])
        print("EPOCH " + str(epoch))
        total_loss = 0
        model_cnn.train()
        model_sem.train()

        labels = []
        for i in range(len(data_docs_cnn)): #the labels are the same for both models
            labels.append(data_docs_cnn[i].label)
        for train, test in kfold.split(np.zeros(4800), labels):
            training_data_cnn = np.array(data_docs_cnn)[train]
            training_data_sem = np.array(data_docs_sem)[train]
            
            test_data_cnn = np.array(data_docs_cnn)[test]
            test_data_sem = np.array(data_docs_sem)[test]
            
            training_data_cnn, training_labels_cnn, train_ids_cnn = data.create_doc_sents(training_data_cnn, 'paragraph', params['task'],
                                                                          params['train_data_limit'])
            training_data_sem, training_labels_sem, train_ids_sem = data.create_doc_sents(training_data_sem, 'paragraph', params['task'],
                                                                          params['train_data_limit'])
            
            test_data_cnn, test_labels_cnn, test_ids_cnn = data.create_doc_sents(test_data_cnn, 'paragraph', params['task'], params['train_data_limit'])
            test_data_sem, test_labels_sem, test_ids_sem = data.create_doc_sents(test_data_sem, 'paragraph', params['task'], params['train_data_limit'])

            steps = int(len(training_data_cnn) / params['batch_size']) #same steps for both
            indices = list(range(len(training_data_cnn)))
            random.shuffle(indices)
            bar = progressbar.ProgressBar()
            for step in bar(range(steps)):

                batch_ind = indices[(step * params["batch_size"]):((step + 1) * params["batch_size"])]
                sentences_cnn, orig_batch_labels = data.get_batch(training_data_cnn, training_labels_cnn, batch_ind, params['model_type'])
                sentences_sem, orig_batch_labels = data.get_batch(training_data_sem, training_labels_cnn, batch_ind, params['model_type'])

                batch_padded_cnn, batch_lengths_cnn, original_index = data.pad_to_batch(sentences_cnn, data.word_to_idx, params['model_type'])
                batch_padded_sem, batch_lengths_sem, original_index = data.pad_to_batch(sentences_sem, data.word_to_idx, params['model_type'])

                model_cnn.zero_grad()
                model_sem.zero_grad()
                if params['model_type'] == 'sem_rel' or params['model_type'] == 'fusion_sem_syn':
                    y_pred = []
                    coherence_pred_cnn = model_cnn(batch_padded_cnn, batch_lengths_cnn, original_index)
                    coherence_pred_cnn_Tensor = coherence_pred_cnn
                    coherence_pred_cnn = coherence_pred_cnn.tolist()
                    
                    coherence_pred_sent, coherence_pred_par = model_sem(batch_padded_sem, batch_lengths_sem, original_index)
                    coherence_pred_sem_sentTensor = coherence_pred_sent
                    coherence_pred_sent = coherence_pred_sent.tolist()
                    
                    coherence_pred_sem_parTensor = coherence_pred_par
                    coherence_pred_par = coherence_pred_par.tolist() 
                                       
                    #gather coherence predictions into one array
                    y_pred.append(coherence_pred_cnn)
                    y_pred.append(coherence_pred_sent)
                    y_pred.append(coherence_pred_par)
                    
                    y_pred = np.array(y_pred)
                    for weights in product(w, repeat=3):
                        if len(set(weights)) == 1:
                            continue
                        weights = normalize(weights)
                        summed = tensordot(y_pred, weights, axes=((0),(0)))
                        # argmax across classes
                        result = argmax(summed, axis=1)
                        # calculate accuracy
                        score = accuracy_score(orig_batch_labels, result)
                        if score > best_score:
                            best_score = score
                            best_weights = weights
                            best_weights = list(best_weights)
                    
                    coherence_pred_cnn = torch.mul(coherence_pred_cnn_Tensor, best_weights[0])
                    coherence_pred_sent = torch.mul(coherence_pred_sem_sentTensor, best_weights[1])
                    coherence_pred_par = torch.mul(coherence_pred_sem_parTensor, best_weights[2])
                    final_prediction1 = coherence_pred_sent.add(coherence_pred_par)
                    final_pred = final_prediction1.add(coherence_pred_cnn)
                    #final_pred = model(batch_padded, batch_lengths, original_index, weights=best_weights)
                    loss_fn = torch.nn.CrossEntropyLoss()
                    loss = loss_fn(final_pred, Variable(LongTensor(orig_batch_labels)))
                    
                mean_loss = loss / params["batch_size"]
                mean_loss.backward()
                total_loss += loss.cpu().data.numpy()
                optimizer_cnn.step()
                optimizer_sem.step()
            fold = 0
            current_time = time.time()
            print("Time %-5.2f min" % ((current_time - timestamp) / 60.0))
            print("Fold" + str(fold) + " - Train loss: " +str(total_loss))
            output_name = params['model_name'] + '_epoch' + str(epoch)
            if params['model_type'] == 'sent_avg' or params['model_type'] == 'par_seq' or params['model_type'] == 'sem_rel' or params['model_type'] == 'cnn_pos_tag' or params['model_type'] == 'fusion_sem_syn':
                
                if params['model_type'] == 'fusion_sem_syn':
                    test_accuracy, test_loss = eval_docs_fusion(model_cnn, model_sem, loss_fn, test_data_cnn, test_data_sem, test_labels_cnn, data, params)                                 
                print("Fold" + str(fold) + " - Test loss: %0.3f" % test_loss)
                if params['task'] == 'score_pred':
                    print("Test correlation: %0.5f" % (test_accuracy))
                else:
                    print("Fold" + str(fold) +" - Test accuracy: %0.2f%%" % (test_accuracy * 100))
            
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                # save best model
                torch.save(model_cnn.state_dict(), params['model_dir'] + '/' + params['model_name'] + '_best')
                torch.save(model_sem.state_dict(), params['model_dir'] + '/' + params['model_name'] + '_best')
                print('saved model ' + params['model_dir'] + '/' + params['model_name'] + '_best')
            print()
            fold += 1
    print("==================== BEST TEST ACCURACY =================================")
    print(best_test_acc)
    return best_test_acc

def test(params, test_docs, data, model):
    if params['model_type'] == 'clique':
        test_data, test_labels = data.create_cliques(test_docs, params['task'])
    elif params['model_type'] == 'sent_avg':
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'sentence', params['task'], params['train_data_limit'])
    elif params['model_type'] == 'par_seq':
        test_data, test_labels, test_ids = data.create_doc_sents(test_docs, 'paragraph', params['task'], params['train_data_limit'])

    if USE_CUDA:
        model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    # output_name = params['model_name'] + '_test'
    if params['model_type'] == 'par_seq' or params['model_type'] == 'sent_avg':
        test_accuracy, test_loss = eval_docs(model, loss_fn, test_data, test_labels, data, params)
        print("Test accuracy: %0.2f%%" % (test_accuracy * 100))
