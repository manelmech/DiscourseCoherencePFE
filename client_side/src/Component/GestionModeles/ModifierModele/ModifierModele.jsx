import React, { useState, useEffect, useCallback } from 'react'
import Container from '@mui/material/Container';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import MenuItem from '@mui/material/MenuItem';
import Slide from '@mui/material/Slide';
import { Alert, AlertTitle } from '@mui/material';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import { Typography } from '@mui/material';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import axios from "axios";
import { LanguageContext } from '../../../context/LangageContext';
import { useContext } from 'react';



const ModifierModele = (props) => {

  const [state, setState] = useState({});
  const [visibilite, setVisibilite] = useState( { visibility: ''});
  const [modelID, setModelID] = useState('')
  const [slideModif, setSlideModif] = useState(null)
  const [modifSuccess, setModifSuccess] = useState(null)
  const [errors, setErrors] = useState({})
  const [slide, setSlide] = useState(null)
  const [annuler, setAnnuler] = useState(null)
  const [confirmer, setConfirmer] = useState(null)
  const { language, changeLanguage } = useContext(LanguageContext);


  const getModele = () => {
        if(props.modele) {
            setState(props.modele)
            setModelID(props.modele.id)
            setVisibilite({visibility : state.visibility})
        }
        console.log(props.modele)
    }
  useEffect(() => {
    getModele();
  }, []);

  // handle fields change
  const handleChange = input => e => {
    if(input == 'visibility') {
        console.log("hello")
        // var vis = (e.target.value === "true");
        // setVisibilite({visibility : vis})
        setState({...state, ['visibility']: e.target.value === 'true'})
      }
    else {
        setState({...state, [input]: e.target.value})
    }  
      
      
      
    //   setState({...state, [vis]: e.target.value})
      console.log(state)
  }

  const handleCloseModif = props.handleCloseModif

  const handleOpenAnnuler = () => {
      setAnnuler(true)
  }

  const handleCloseAnnuler = () => {
      setAnnuler(false)
  }

  const handleOpenConfirmer = () => {
      setConfirmer(true)
  }

  const handleCloseConfirmer = () => {
      setConfirmer(false)
  }

  const validate = (fieldValues = state) => {
      console.log(fieldValues)
      let temp = { ...errors }
      if ('name' in fieldValues)
          temp.name = fieldValues.name ? "" :  language=="fr"? "Ce champs est requis."  : "This field is required."

      if ('description' in fieldValues)
          temp.description = fieldValues.description ? "" : language=="fr"? "Ce champs est requis."  : "This field is required."
      if ('accuracy' in fieldValues)
          temp.accuracy = fieldValues.accuracy ? "" : language=="fr"? "Ce champs est requis."  : "This field is required."
      if ('precision' in fieldValues)
          temp.precision = fieldValues.precision ? "" : language=="fr"? "Ce champs est requis."  : "This field is required."
      if ('rappel' in fieldValues)
          temp.rappel = fieldValues.rappel ? "" : language=="fr"? "Ce champs est requis."  : "This field is required."
      if ('F1_score' in fieldValues)
          temp.F1_score = fieldValues.F1_score ? "" : language=="fr"? "Ce champs est requis."  : "This field is required."
    //   if ('visibility' in fieldValues)
      temp.visibility = ""
      setErrors({
          ...temp
      })

    //   if (fieldValues == state)
          return Object.values(temp).every(x => x == "")
  }
    
  const message = (
      <div style={{margin:'10px 40px 30px 40px'}}>
          <Slide direction="up" in={slide} mountOnEnter unmountOnExit>
          <Alert severity="error">
              <strong></strong>
              {language=="fr"? "Veuillez renseigner les champs requis." : "Please fill in the required fields."}

          </Alert>
          </Slide>
      </div>
  )

  
  const modifSuccessMessage = (
      <div style={{margin:'20px 0px', padding:'12px'}}>
                {(modifSuccess == true) && (
                  <Slide direction="up" in={slideModif} mountOnEnter unmountOnExit>
                  <Alert severity="success" onClose={() => {
                      setSlideModif(false)
                      }}>
                      <AlertTitle>{language=="fr"? "Succés" : "Success"}</AlertTitle>
                       {language=="fr"? "Le modèle a été modifié" : "The model has been modified"} <strong>{language=="fr"? "avec succés" : "with success"}</strong>
                  </Alert>
                  </Slide>
                ) } { (modifSuccess == false) && (
                  <Slide direction="up" in={slideModif} mountOnEnter unmountOnExit>
                  <Alert severity="error">
                      <AlertTitle>{language=="fr"? "Erreur!" : "Error!"}</AlertTitle>
                      <strong>{language=="fr"? "Erreur lors de la modification du modèle" : "Error while modifying the template"}</strong>
                  </Alert>
                  </Slide>
                ) }
      </div>
    )
    
    const onModifierModele = useCallback( 
      async () => {
        window.setTimeout( function(){
                            handleCloseConfirmer()
                            setModifSuccess(null)
                            window.location.href = "/gestionmodeles";
                          }, 2000 );
        const response = await axios.put(`http://localhost:8080/update_model/${modelID}`, {
            "id": state.id,
            "name": state.name,
            "description": state.description,
            "saved_model_pickle": state.saved_model_pickle,
            "hybridation": state.hybridation,
            "preprocess": state.preprocess,
            "F1_score": state.F1_score,
            "precision": state.precision,
            "rappel": state.rappel,
            "accuracy": state.accuracy,
            "visibility" : state.visibility
          })
            .then((response) => {
                console.log(state)
                setSlideModif(true)
                setModifSuccess(true)
                console.log("modifié")
                console.log(response);
                console.log(state)
                window.setTimeout( function(){
                    handleCloseConfirmer()
                    setModifSuccess(null)
                    window.location.href = "/gestionmodeles";
                }, 2000 );
                }, (error) => {
                setSlideModif(true)
                setModifSuccess(false)
                console.log("erreur")
                console.log(error);
                window.setTimeout( function(){
                    handleCloseConfirmer()
                    setModifSuccess(null)
                }, 2000 );
                });
      });

    const continuer = e => {
        e.preventDefault();
        if(validate()){
            console.log(state)
            setSlide(null)
            handleOpenConfirmer()
        } else {
            setSlide(true)
        }
    }

    const annulerDialogue = (
        <div>
            <Dialog
                open={annuler}
                aria-labelledby="alert-dialog-title"
                aria-describedby="alert-dialog-description"
            >
            <Typography style={{fontFamily:'Poppins', fontSize:'18px', padding:'14px 20px', boxShadow:'none'}}>
                    
                    {language=="fr"? "Voulez-vous vraiment annuler la modification du modèle? " : "Are you sure you want to cancel the template modification?"}

                    <br></br>
                    {language=="fr"? "Toutes les informations saisies seront perdues." : "All information entered will be lost."}

            </Typography>                    
                <DialogActions>
                <Button onClick={handleCloseModif} style={{textTransform:"capitalize", color:"#F5365C", fontFamily:'Poppins', margin:"12px 20px", fontWeight:"bold"}}>
               {language=="fr"? "Oui" : "Yes"}

                </Button>
                <Button onClick={handleCloseAnnuler} style={{textTransform:"capitalize", backgroundColor:"#252834", color:"white", fontFamily:'Poppins', padding:"6px 12px", margin:"12px 20px"}}>
                {language=="fr"? "Non" : "No"}

                </Button>
                </DialogActions>
            </Dialog>
        </div>
    )

    const confirmerDialogue = (
        <div>
            <Dialog
                open={confirmer}
                aria-labelledby="alert-dialog-title"
                aria-describedby="alert-dialog-description"
            >
            <Typography style={{fontFamily:'Poppins', fontSize:'17px', padding:'20px 17px 18px', boxShadow:'none'}}>
                    {language=="fr"? "Voulez-vous confirmer la modification du modèle? " : "Do you want to confirm the model change?"}


                </Typography>                    
                <DialogActions>
                <Button onClick={onModifierModele} style={{textTransform:"capitalize", color:"#F5365C", fontFamily:'Poppins', margin:"12px 12px", fontWeight:"bold"}}>
                   {language=="fr"? "Oui " : "Yes"}

                </Button>
                <Button onClick={handleCloseConfirmer} style={{textTransform:"capitalize", backgroundColor:"#252834", color:"white", fontFamily:'Poppins', padding:"6px 12px", margin:"12px 20px"}}>
                {language=="fr"? "Non " : "No"}
                </Button>
                </DialogActions>
                {modifSuccess ? modifSuccessMessage : <br></br>}
            </Dialog>
        </div>
    )

    const visibilities = [
        {
            label: "Oui",
            value: true,
        },
        {
            label: "Non",
            value: false,
        },
    ]
    
        return (
             <Container fluid style={{paddingBottom:"40px"}}>
                    <div style={{padding:"10px", fontFamily:"Poppins"}}>
                    <h5 style={{margin:"10px 15% 9px" , justifyContent:'center'}}> {language=="fr"? "Modifier les informations du modèle" : "Edit model information"}
</h5>
                    </div>
                    <form noValidate="false">
                        <div style={{padding:"5px 40px"}}>
                          {/* <InputLabel>Nom du modèle</InputLabel> */}
                          <TextField
                              required
                              error={errors.name === "" ? false : ""}
                              id="name"
                              variant="outlined"
                              label= {language=="fr"? "Nom du modèle": "Model name"}

                              InputLabelProps={{
                                shrink: true,
                              }}
                              fullWidth='true'
                              value={state.name ?? ''}
                              onChange={handleChange('name')}
                          />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                          <TextField
                              required
                              error={errors.description === "" ? false : ""}
                              id="description"
                              variant="outlined"
                              label="Description"
                              InputLabelProps={{
                                shrink: true,
                              }}
                              fullWidth='true'
                              value={state.description ?? ''}
                              onChange={handleChange('description')}
                          />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                            <InputLabel id="demo-simple-select-label">{language=="fr"? "Visibilité" : "Visibility"}  </InputLabel>
                            <select value={state.visibility ?? ''} onChange={handleChange('visibility')}>
                                {visibilities.map((option) => (
                                <option value={option.value}>{option.label}</option>
                                ))}
                            </select>
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                          <TextField
                                required
                                id="accuracy"
                                variant="outlined"
                                label={language=="fr"? "Exactitude": "Accuracy"}
                                InputLabelProps={{
                                  shrink: true,
                                }}
                                fullWidth='true'
                                value={state.accuracy ?? ''}
                                onChange={handleChange('accuracy')}
                            />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                          <TextField
                                required
                                error={errors.precision === "" ? false : ""}
                                id="precision"
                                variant="outlined"
                                label={language=="fr"? "Précision": "Precision"}
                                InputLabelProps={{
                                  shrink: true,
                                }}
                                fullWidth='true'
                                value={state.precision ?? ''}
                                onChange={handleChange('precision')}
                              />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                          <TextField
                              required
                              error={errors.rappel === "" ? false : ""}
                              id="rappel"
                              variant="outlined"
                              label={language=="fr"? "Rappel": "Recall"}
                              InputLabelProps={{
                                shrink: true,
                              }}
                              fullWidth='true'
                              onChange={handleChange('rappel')}
                              value={state.rappel ?? ''}
                            />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                          <TextField
                                required
                                error={errors.F1_score === "" ? false : ""}
                                id="F1_score"
                                variant="outlined"
                                label={language=="fr"? "Score F1": "F1 score"}
                                InputLabelProps={{
                                  shrink: true,
                                }}
                                fullWidth='true'
                                value={state.F1_score ?? ''}
                                onChange={handleChange('F1_score')}
                            />
                        </div>
                        {message}
                        <div className="flex-container" style={{display: "flex", flexWrap:'wrap', gap:'30px', justifyContent:'center', alignItems:'center'}}>
                                <div>
                                <Button onClick={handleOpenAnnuler} style={{backgroundColor:"white", textTransform:"capitalize", color:"#5885FB", fontWeight:'bold'}} variant="contained">
                    
                                    {language=="fr"? " Annuler " : "Cancel"}

                                </Button>
                                </div>
                                <div>
                                <Button onClick={continuer} style={{backgroundColor:"#5885FB", textTransform:"capitalize", color:"white", fontWeight:'bold', width:'150px'}} variant="contained">
                                   
                                    {language=="fr"? " Sauvegarder " : "Save"}

                                </Button>
                                </div>
                        </div>
                        {annulerDialogue}
                        {confirmerDialogue}

                    </form>
            </Container>
        )
  }

export default ModifierModele