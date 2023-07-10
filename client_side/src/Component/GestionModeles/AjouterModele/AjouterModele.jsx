import React, { useState } from 'react'
import Container from '@mui/material/Container';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import './AjouterModele.css'
import Slide from '@mui/material/Slide';
import { Alert, AlertTitle,FormControl } from '@mui/material';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import FileUploadOutlinedIcon from '@mui/icons-material/FileUploadOutlined';

import {MenuItem ,Select,InputLabel} from '@mui/material';
import { Typography } from '@mui/material';
import axios from 'axios'
import Link from '@mui/material/Link';
import { styled } from '@mui/material/styles';
import { LanguageContext } from '../../../context/LangageContext';
import { useContext } from 'react';



export const AjouterModele = ({handleCloseAjout}) => {

    const [state, setState] = useState({
        id: 55,
        name: '',
        preprocess : '',
        description: '',
        accuracy: '',
        precision: '',
        rappel: '',
        F1_score: '',
      
        visibility: true,
       
    });

    
    const [errors, setErrors] = useState({})
    const [slide, setSlide] = useState(null)
    const [slideErr, setSlideErr] = useState(null)
    const [annuler, setAnnuler] = useState(null)
    const [success, setSuccess] = useState(false)
    const [loading, setLoading] = useState(false);
    const [detailError, setDetailError] = useState("");
    const { name, description, accuracy, precision, rappel, F1_score, preprocess } = state;
    const values = { name, description, accuracy, precision, rappel, F1_score, preprocess };
    
    const [selectedFile, setSelectedFile] = useState(null);
    const [selectedFileName, setselectedFileName] = useState('');

    const [selectedFileConfig, setSelectedFileConfig] = useState(null);
    const [selectedFileNameConfig, setselectedFileNameConfig] = useState('');
    const { language, changeLanguage } = useContext(LanguageContext);






    // handle fields change
    const handleChange = input => e => {
        setState({...state, [input]: e.target.value
        })
    }


    const handleOpenAnnuler = () => {
        setAnnuler(true)
    }

    const handleCloseAnnuler = () => {
        setAnnuler(false)
    }

    const validate = (fieldValues = values) => {
        let temp = { ...errors }
        if ('name' in fieldValues)
            temp.name = fieldValues.name ? "" : "Ce champs est requis."
        if ('description' in fieldValues)
            temp.description = fieldValues.description ? "" : "Ce champs est requis."
        if ('accuracy' in fieldValues)
            temp.accuracy = fieldValues.accuracy ? "" : "Ce champs est requis."
        if ('precision' in fieldValues)
            temp.precision = fieldValues.precision ? "" : "Ce champs est requis."
        if ('rappel' in fieldValues)
            temp.rappel = fieldValues.rappel ? "" : "Ce champs est requis."
        if ('F1_score' in fieldValues)
            temp.F1_score = fieldValues.F1_score ? "" : "Ce champs est requis."
        if ('preprocess' in fieldValues)
            temp.preprocess = fieldValues.preprocess ? "" : "Ce champs est requis."
        
            // "hybridation": hybridation,
        setErrors({
            ...temp
        })
         
        if (fieldValues == values && selectedFile){
            return Object.values(temp).every(x => x == "")
        }
        
           
            else{
              
              return false
            }
        
    }
  
    
    const validateFileOption = () =>{
        if (selectedFileConfig){
            return true
    }
    else{
        return false
    }
}
    const message = (
        <div style={{margin:'10px 40px 30px 40px'}}>
            <Slide direction="up" in={slideErr} mountOnEnter unmountOnExit>
                <Alert severity="error">
                {detailError!=="" ? (
          <strong>{detailError}</strong>
        ) : (
          <strong> {language=="fr"? "Veuillez renseigner les champs requis." : "Please fill in the required fields."}

            </strong>
        )}
                </Alert>
            </Slide>
        </div>

    )

    const successMessage = (
        <div style={{margin:'10px 40px 30px 40px'}}>
            {success && (
                <Slide direction="up" in={slide} mountOnEnter unmountOnExit>
                    <Alert severity="success">
                        <AlertTitle>{language=="fr"? "Succés" : "Success"}</AlertTitle>
                        {language=="fr"? "Le modèle a été ajouté" : "The model has been added"} <strong>{language=="fr"? "avec succés" : "with success"} </strong>
                    </Alert>
                </Slide>
                ) } { !success && (
                <Slide direction="up" in={slide} mountOnEnter unmountOnExit>
                    <Alert severity="error">
                        <AlertTitle>{language=="fr"? "Erreur!" : "Error!"}</AlertTitle>
                        <strong>{language=="fr"? "Le modèle n'a pas été ajouté avec succés" : "Model was not added successfully"}</strong>
                    </Alert>
                </Slide>
            ) }
        </div>

    )

    const annulerDialogue = (
        <div>
            <Dialog
                open={annuler}
                aria-labelledby="alert-dialog-title"
                aria-describedby="alert-dialog-description"
            >
                <Typography style={{fontFamily:'Poppins', fontSize:'15px', padding:'14px 20px', boxShadow:'none'}}>
                    {language=="fr"? " Voulez-vous vraiment annuler l'ajout d'un nouveau modèle? " : "Are you sure you want to cancel adding a new model?"}
                    <br></br>
                    {language=="fr"? "Toutes les informations saisies seront perdues. " : "All information entered will be lost."}

                </Typography>                    
                <DialogActions>
                    <Button onClick={handleCloseAjout} style={{textTransform:"capitalize", color:"#F5365C", fontFamily:'Poppins', margin:"12px 20px", fontWeight:"bold"}}>
                    {language=="fr"? "Oui " : "Yes"}
                    </Button>
                    <Button onClick={handleCloseAnnuler} style={{textTransform:"capitalize", backgroundColor:"#252834", color:"white", fontFamily:'Poppins', padding:"6px 12px", margin:"12px 20px"}}>
                    {language=="fr"? "Non " : "No"}

                    </Button>
                </DialogActions>
            </Dialog>
        </div>
    )




      const [inputType, setInputType] = useState('manual');

      const handleInputChange = (event) => {
        setInputType(event.target.value);
      };

    const Input = styled('input')({
        display: 'none',
      });

  



  
    const changeHandler = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
        setselectedFileName(file ? file.name : '');
       
    };

    const changeHandlerConfig = (event) => {
        const file = event.target.files[0];
        setSelectedFileConfig(file);
        setselectedFileNameConfig(file ? file.name : '');
       
    };
    const handleSubmit = async  (event) => {
      event.preventDefault();
      
        setLoading(true)
        const formData2 = new FormData();
        if (inputType==="manual")
        {
            if(validate()){
                const model = {
                    "id": 55,
                    "name" : name,
                    "description" : description,
                    "F1_score": F1_score,
                    "precision" : precision,
                    "accuracy" : accuracy,
                    "rappel": rappel,
                    "preprocess" : preprocess,
                    "hybridation": false,
                    "visibility": true,
                    "cloud":false
                  };
                  
                  for (const [key, value] of Object.entries(model)) {
                    formData2.append(key, value);
                  }
                  formData2.append(
                    'file_info',
                    selectedFile
                    
                  );
        
                  await axios.post('http://localhost:8080/add_model', formData2).then((response) => {
                    setLoading(false)
                    setSlide(true)
                     setSuccess(true)
                     setSlideErr(false)
                     console.log(response);
                     window.setTimeout( function(){
                         window.location.href = "/gestionmodeles";
                     }, 2000 );
                   }, (error) => {
                     
                     setLoading(false)
                     setSlideErr(true)
                     setSuccess(false)
                     //setDetailError(error.response.data.detail)
                     
                   })
            }
            else {
                setDetailError("")
                setSlideErr(true)
                setLoading(false)
            }
       
        }
  
        else{
            if(validateFileOption()){
                formData2.append(
                    'file_config',
                    selectedFileConfig
                    
                  );
              
        
                  await axios.post('http://localhost:8080/add_model_file_option', formData2).then((response) => {
                    setLoading(false)
                    setSlide(true)
                     setSuccess(true)
                     setSlideErr(false)
                     console.log(response);
                     window.setTimeout( function(){
                         window.location.href = "/gestionmodeles";
                     }, 2000 );
                   }, (error) => {
                     setLoading(false)
                     setSlideErr(true)
                     setSuccess(false)
                     setDetailError(error.response.data.detail)
                   })
            }

            else{
                setDetailError("")
                setSlideErr(true)
                setLoading(false)
            }
       
        }
     
   
    
      
     

         
             
              
    }
    
  

  
    

    return (
             <Container fluid style={{paddingBottom:"40px"}}>
                    <div style={{padding:"10px", fontFamily: 'Poppins'}}>
                        <h4 style={{margin:"10px 25% 10px"}}>Nouveau modèle</h4>
                    </div>

                    <div style={{padding:"5px 40px"}}>
                  
                    <FormControl fullWidth>
        <InputLabel id="demo-simple-select-label">
        {language=="fr"? "Choisir une option pour entrer les paramètres du modèles" : "Choose an option to enter Model parameters"}

        </InputLabel>
        <Select
    labelId="demo-simple-select-label"
    id="demo-simple-select"
    value={inputType}
   label="Choisir une option pour entrer les paramètres du modèles"
   onChange={handleInputChange}
  >
    <MenuItem value="manual">{language=="fr"? "Entrer les paramètres manuellement " : "Enter parameters manually"}
</MenuItem>
    <MenuItem value="file">{language=="fr"? "Entrer les paramètres dans un fichier json" : "Enter parameters in a json file"}</MenuItem>
   
  </Select>
  </FormControl>
                        </div> <br></br>
                    <form noValidate="false">
                 
                   {inputType==="manual" && (<div><div style={{padding:"5px 40px"}}>
                            <TextField
                                required
                                error={errors.name === "" ? false : ""}
                                id="name"
                                label="Nom du modèle"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('name')}
                                value={values.name}
                                type='string'
                            />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                            <TextField
                                required
                                error={errors.preprocess === "" ? false : ""}
                                id="preprocess"
                                label="Niveau de prétraitement"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('preprocess')}
                                value={values.preprocess}
                                type='string'
                            />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                            <TextField
                                required
                                error={errors.description === "" ? false : ""}
                                id="description"
                                label="Description"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('description')}
                                defaultValue={values.description}
                                type='string'
                            />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                            <TextField
                                required
                                error={errors.accuracy === "" ? false : ""}
                                id="accuracy"
                                label="Exactitude"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('accuracy')}
                                defaultValue={values.accuracy}
                                type='string'
                            />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                            <TextField
                                required
                                error={errors.precision === "" ? false : ""}
                                id="precision"
                                label="Précision"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('precision')}
                                defaultValue={values.precision}
                                type='string'
                            />
                        </div>
                        <br></br> 
                        <div style={{padding:"5px 40px"}}>
                            <TextField
                                required
                                error={errors.rappel === "" ? false : ""}
                                id="rappel"
                                label="Rappel"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('rappel')}
                                defaultValue={values.rappel}
                            />
                        </div>
                        <br></br>
                        <div style={{padding:"5px 40px"}}>
                            <TextField
                                required
                                error={errors.F1_score === "" ? false : ""}
                                id="F1_score"
                                label="Score F1"
                                InputLabelProps={{
                                    shrink: true,
                                }}
                                variant="outlined"
                                fullWidth='true'
                                onChange={handleChange('F1_score')}
                                defaultValue={values.F1_score}
                                type='string'
                            />
                        </div>
                        <br></br>               <div style={{padding:"5px 40px"}}>
                        <p>{language=="fr"? "Entrer le fichier pickle*" : "Enter pickle file*"}</p>
                        <label htmlFor="file_input" className="custom-file-upload">
                        <FileUploadOutlinedIcon className='icon-import'/> 
                        {selectedFileName || 'Importer le fichier'}
                         </label>
                        <input  id="file_input" type="file" name="image" onChange={changeHandler}/>
                        </div> </div>) }     
                        {inputType==="file" && (   <div style={{padding:"5px 40px"}}>
                            <p>{language=="fr"? "Entrer le fichier des paramètres*" : "Enter settings file*"}</p>
                        <label htmlFor="file_input_config" className="custom-file-upload">
                        <FileUploadOutlinedIcon className='icon-import'/> 
                        {selectedFileNameConfig || 'Importer le fichier'}
                       
                         </label>
                        <input  id="file_input_config" type="file" name="image" onChange={changeHandlerConfig}/>
                        </div>)}



          
                        {
                    loading == true ? 
                    <div style={{position: "relative", margin: "0px 50%"}}>
                    <CircularProgress/> 
                    </div>
                    : <></>
                    }
                        {message}
                        {successMessage}
                        <div className="flex-container" style={{display: "flex", flexWrap:'wrap', gap:'30px', justifyContent:'center', alignItems:'center'}}>
                            <div>
                                <Button onClick={handleOpenAnnuler} style={{backgroundColor:"white", textTransform:"capitalize", color:"#5885FB", fontWeight:'bold'}} variant="contained">
                              {language=="fr"? "Annuler" : "Cancel"}
                                </Button>
                            </div>
                        <div>

                        <Button onClick={handleSubmit} style={{backgroundColor:'#5885FB', textTransform:"capitalize", color:"white", fontWeight:'bold', width:'150px'}} variant="contained">
                            <Link to="/gestionmodeles" variant="contained" style={{fontFamily:'Poppins', color:'white'}}>
                                {language=="fr"? "Confirmer" : "Confirm"}

                                
                            </Link>
                            </Button>
                        </div>
                        </div>
                        {annulerDialogue}
                    </form>
            </Container>
    )
}

export default AjouterModele