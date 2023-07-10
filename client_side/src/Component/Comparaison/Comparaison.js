import React, { useState, useEffect, useCallback, Suspense, lazy } from 'react';
import axios from 'axios'
import { Form, Row, Col, Stack } from "react-bootstrap";
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import ClickAwayListener from '@mui/material/ClickAwayListener';
import Grow from '@mui/material/Grow';
import Paper from '@mui/material/Paper';
import Popper from '@mui/material/Popper';
import MenuItem from '@mui/material/MenuItem';
import MenuList from '@mui/material/MenuList';
import './Comparaison.css'
// import Sidebar from '../Sidebar/Sidebar'
import ResultComparaisonOne from '../ResultComparaisonOne/ResultComparaisonOne'
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import ThreeSixtyIcon from '@mui/icons-material/ThreeSixty';
import Typography from '@mui/material/Typography';
import LooksOneIcon from '@mui/icons-material/LooksOne';
import LooksTwoIcon from '@mui/icons-material/LooksTwo';
import{ReactToPrint} from 'react-to-print'
import Multiselect from "multiselect-react-dropdown";
import ApexCharts from 'apexcharts';
import Dialog from '@mui/material/Dialog';

import FileUploadOutlinedIcon from '@mui/icons-material/FileUploadOutlined';
const Sidebar = React.lazy(() => import('../Sidebar/Sidebar'));

const Comparaison  = () => {

  const [csv, setcsv] = useState(false);
  const [text, setText] = useState("");
  const [open, setOpen] = React.useState(false);
  const anchorRef = React.useRef(null);
  const hiddenFileInput = React.useRef(null);
  const hiddenFile = React.useRef(null);

  
  const [index, setIndex] = React.useState(0);
  const [isLoading, setLoading] = useState(null)
  const [isEmpty, setEmpty] = useState(true)
  const [state, setState] = useState(null)
  const [metrics, setMetrics] = useState(null)

  const [datacomp, setDatacomp] = useState(null)
  const [data, setData] = useState(null)

  const [scoreResult, setScore] = useState(null);
  const [chartLength, setChartLength] = useState(0);
  const [modelsLength, setModelsLength] = useState(0);
  const [array_cell, setArray] = useState(null);
  const [fileName, setFileName] = useState(null)
  const [modeles, setModeles] = useState(null)
  const [descriptionList, setDescriptionList] = useState({})
  const [modelNames, setModelNames] = useState([])
  const [visibleModels, setVisibleModels] = useState([])
  const [modelIDs, setModelIDs] = useState(null)
  const [selectedIndex, setSelectedIndex] = useState(null)
  const [performances, setPerformances] = useState([])
  const [accuracy, setAccuracy] = useState()
  const [precision, setPrecision] = useState()
  const [rappel, setRappel] = useState()
  const [f1_score, setF1score] = useState()
  const [files, setFiles]=useState([])
  const [modelList, setmodelList] = useState(["sent_avg"]);
  const [indexes, setindexes] = useState([]);
  const indexlist = [];
  const [openDialog,setOpenDialog] = useState(false)
  const [selectedOption, setSelectedOption] = useState('');



 

  function selectProps(...props) {
    return function (obj) {
      const newObj = {};
      props.forEach(name => {
        newObj[name] = obj[name];
      });
      return newObj;
    }
  }

  useEffect(() => {
    axios.get('http://localhost:8080/models').then(async (res) => {
      const { data } = res;
      console.log(data)
      if (data) {
        setModeles(data);
        let visible = data.filter(obj => {
          return obj.visibility === true
        })
        console.log(visible)
        const descriptions = visible.map(selectProps("description"));
        var temp_desc = descriptions.map(Object.values);
        const ids = visible.map(selectProps("id"));
        var temp = ids.map(Object.values);
        setDescriptionList(temp_desc.flat(1))
        setModelIDs(temp.flat(1))
        console.log(temp)

        const accuracy_const = visible.map(selectProps("accuracy"));
        var temp_acc = accuracy_const.map(Object.values);
        setAccuracy(temp_acc.flat(1))

        const precision_const = visible.map(selectProps("precision"));
        var temp_prec = precision_const.map(Object.values);
        setPrecision(temp_prec.flat(1))

        const rappel_const = visible.map(selectProps("rappel"));
        var temp_rapp = rappel_const.map(Object.values);
        setRappel(temp_rapp.flat(1))

        const F1_score_const = visible.map(selectProps("F1_score"));
        var temp_F1 = F1_score_const.map(Object.values);
        setF1score(temp_F1.flat(1))

        const names = visible.map(selectProps("name"));
        var temp = names.map(Object.values);
        setModelNames(temp.flat(1))
      }
    })
  }, [data]);

  function createData(text, modelname, predicted_score) {
    return { text, modelname, predicted_score};
  }

  function createDatacsv(text_id, text, original_score, predicted_score, modelname) {
    return { text_id, text, original_score, predicted_score ,modelname};
  }

 
  const handleSubmit = (event) => {
    setLoading(true)
    event.preventDefault();
    const params = { text, modelList };

    var divelement = document.getElementById('evalSection')
    setModelsLength(modelList.length)
    if (files.length == 0) {
      
      
      console.log(params)
      axios
        .post('http://localhost:8080/evaluatecomp/', params)
        .then((res) => {
          const data = res.data.data
          const score = data.scores
          const names = data.modelnames
          
          let table_details = []
          let chart_result = []
          setChartLength(score.length)
          let index = 0
          while (index < score.length) {
            score[index]++;
            index++;
          }
          //alert(score)
          // var myArray = JSON.Parse(score);
          for (var i = 0; i < modelList.length; i++) {
        
            chart_result.push(
              {
                label: modelList[i],                 
              }
              )          
               
    
               let count_low = 0
               let count_med = 0
               let count_high = 0
               
            
                if (data.scores[i]  == 1) {
                  count_low++
                } else if (data.scores[i]  == 2) {
                  count_med++
                } else {
                  count_high++
                }
    
          chart_result[i]["Low"]= count_low 
          chart_result[i]["Medium"]= count_med
          chart_result[i]["High"]= count_high
    
        }

        for (var i = 0; i < score.length; i++) {
        
          let cell = createData(data.texts[i], data.modelnames[i], data.scores[i])
          table_details.push(cell)
    
        }

          setChartLength(Object.keys(chart_result[0]).length-1)
          setArray(table_details)
          setState(chart_result)
          setEmpty(false)
          setLoading(false)

        })
        .catch((error) => {
          alert(`Error: ${error.message}`)
          divelement.hidden = false
          setScore(error.message)
          setLoading(false)
        })
    } else {

      console.log(files)
      
      const data = new FormData()
      files.forEach(file =>{
        data.append('data',file)
        console.log(file)})
      
      modelList.forEach(model =>{
          data.append('modelnames',model)
          console.log(model)})
      data.append('csv',csv)
      
      
   

      /*axios
      .post('http://localhost:8080/uploadcsvcomp',datacomp)
      .then((res) => {

        const data = res.data.data
        const score = data.scores
        let chart_result=[]
        let table_details = []
        setChartLength(score.length)

        
        
        for (var i = 0; i < modelList.length; i++) {
      
          chart_result.push(
            {
              label: modelList[i],
              Accuracy: data.Accuracy[i],
              precision: data.Precision[i] ,
              Recall: data.Recall[i],
              F1_score: data.F1score[i],

                   
            })}
     
    for (var i = 0; i < score.length; i++) {
      let cell = createDatacsv( data.text_ids[i], data.texts[i], data.original_scores[i],data.scores[i],data.modelnames[i])
      table_details.push(cell)

    }

        console.log(table_details)
        console.log(chart_result)
        setArray(table_details)
        setState(chart_result)
        setEmpty(false)
        setLoading(false)
      })
      .catch((error) => {
        alert(`Error: ${error.message}`)
        divelement.hidden = false
        setScore(error.message)
        setLoading(false)
      })  */
    axios
    .post('http://localhost:8080/uploadfilecomp',data)
    .then((res) => {

      const data = res.data.data
      const score = data.scores
      const names = data.modelnames
      
      if( !csv){
      /*let table_details = []
      setChartLength(score.length)
      let index = 0
      while (index < score.length) {
        score[index]++;
        index++;
      }
      //alert(score)
      // var myArray = JSON.Parse(score);
      let chart_result = []

      for (var i = 0; i < score.length; i++) {
        
        let cell = createData(data.texts[i], data.modelnames[i], data.scores[i])
        table_details.push(cell)

      }

      for (var i = 0; i < modelList.length; i++) {
        
        chart_result.push(
          {
            label: modelList[i],                 
          }
          )
           
           var k=0
           var l=modelList.length
           let j = i
          while(j < score.length){
          chart_result[i]["Text"+k]= data.scores[j] 
          j=j+l
          k=k+1
        }

      }*/ 
      let table_details = []
      let chart_result=[]

      let index = 0
      while (index < score.length) {
        score[index]++;
        index++;
      }
      //alert(score)
      // var myArray = JSON.Parse(score);
      for (var i = 0; i < modelList.length; i++) {
        
        chart_result.push(
          {
            label: modelList[i],                 
          }
          )
           
           var k=0
           var l=modelList.length
           let j = i

           let count_low = 0
           let count_med = 0
           let count_high = 0
           
          while(j < score.length){

            if (data.scores[j]  == 1) {
              count_low++
            } else if (data.scores[j]  == 2) {
              count_med++
            } else {
              count_high++
            }

            j=j+l
            k=k+1
      }
      chart_result[i]["Low"]= count_low 
      chart_result[i]["Medium"]= count_med
      chart_result[i]["High"]= count_high

    }

    for (var i = 0; i < score.length; i++) {
        
      let cell = createData(data.texts[i], data.modelnames[i], data.scores[i])
      table_details.push(cell)

    }


      setChartLength(Object.keys(chart_result[0]).length-1)
      console.log(chart_result)
      setArray(table_details)
      setState(chart_result)
      setEmpty(false)
      setLoading(false)}




      else{
        let table_details = []
        let chart_result=[]
  
        let index = 0
        while (index < score.length) {
          score[index]++;
          index++;
        }
        //alert(score)
        // var myArray = JSON.Parse(score);
        for (var i = 0; i < modelList.length; i++) {
          
          chart_result.push(
            {
              label: modelList[i],                 
            }
            )
             
             var k=0
             var l=modelList.length
             let j = i
  
             let count_low = 0
             let count_med = 0
             let count_high = 0
             
            while(j < score.length){
            chart_result[i]["Text"+k]= data.scores[j] 
  
              if (data.scores[j]  == 1) {
                count_low++
              } else if (data.scores[j]  == 2) {
                count_med++
              } else {
                count_high++
              }
  
              j=j+l
              k=k+1
        }
        chart_result[i]["Low"]= count_low 
        chart_result[i]["Medium"]= count_med
        chart_result[i]["High"]= count_high
  
      }
  


        let chart_metrics=[]
       
     

              
        for (var i = 0; i < modelList.length; i++) {
      
          chart_metrics.push(
            {
              label: modelList[i],
              Accuracy: data.Accuracy[i],
              precision: data.Precision[i] ,
              Recall: data.Recall[i],
              F1_score: data.F1score[i],

                   
            })}
     
    for (var i = 0; i < score.length; i++) {
      let cell = createDatacsv( data.text_ids[i], data.texts[i], data.original_scores[i],data.scores[i],data.modelnames[i])
      console.log(cell)
      table_details.push(cell)

    }

        console.log(table_details)
        setMetrics(chart_metrics)
        setArray(table_details)
        setState(chart_result)
        setEmpty(false)
        setLoading(false)

      }
    })
    .catch((error) => {
      alert(`Error: ${error.message}`)
      divelement.hidden = false
      setScore(error.message)
      setLoading(false)
    })
    
  
    } }
     


  
    

 


  const handleFileInputChange = (event) =>{
   
    setFiles(Array.from(event.target.files))
    console.log(files)
    setOpenDialog(false);
  }


  const handleClickOpenDialog = () => {
    setOpenDialog(true);
  };
  
  
 


  const handleImportcsv = event => {
    var textarea = document.getElementById('CheckIt');
    textarea.required = false;
    textarea.disabled = true;
    hiddenFileInput.current.click();
    setcsv(true)

  };


  const handleImporttxt = event => {
    var textarea = document.getElementById('CheckIt');
    textarea.required = false;
    textarea.disabled = true;
    hiddenFile.current.click();

  };

  const handleChange = event => {
    const fileUploaded = event.target.files[0];
    if (fileUploaded) {
      let dataFile = new FormData();
      dataFile.append('file', fileUploaded);
      setDatacomp(dataFile)
      setFiles(dataFile)
      setFileName(fileUploaded.name)
      setOpenDialog(false);
    }
  };


  const handleRefresh = () => {
      
   

// Join line array back into a string
  };

  const handleTestChange = (event) => {
    setSelectedOption(event.target.value);
    console.log(event.target.value);

    if( event.target.value==="option1"){
          // Split the text into sentences using regular expressions
      const regex = /([^.!?]+[.!?]+)/g;
      const sentences = text.match(regex);

      // Shuffle the array of sentences using the Fisher-Yates algorithm
      for (let i = sentences.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [sentences[i], sentences[j]] = [sentences[j], sentences[i]];
      }

      // Join the shuffled sentences back together while preserving punctuation
      const shuffledText = sentences.join(' ');
          // Join the shuffled sentences back into a text
      document.getElementById("CheckIt").value= shuffledText;
      
      setText(shuffledText)


}
if( event.target.value==="option2"){
      const textArray = text.split(" "); // Split the text into an array of words

      // Generate random indices to determine the start and end of the part to be deleted
      const startIndex = Math.floor(Math.random() * textArray.length);
      const endIndex = Math.floor(Math.random() * textArray.length);
    
      // Ensure the start index is smaller than the end index
      const [start, end] = [startIndex, endIndex].sort((a, b) => a - b);
    
      // Remove the words between the start and end indices
      const deletedPart = textArray.splice(start, end - start + 1).join(" ");
  
  
      // Join the shuffled sentences back into a text
     document.getElementById("CheckIt").value= deletedPart;
      
     setText(deletedPart)


    }

  };




  const handleClick = () => {
  };

  const handleMenuItemClick = (event, index) => {
    setIndex(index);
    setSelectedIndex(modelIDs[index])
    setOpen(false);
  };

  const handleToggle = () => {
    setOpen((prevOpen) => !prevOpen);
  };

  const handleClose = (event) => {
    if (anchorRef.current && anchorRef.current.contains(event.target)) {
      return;
    }
    
    
    
  
  

    setOpen(false);
  };
  const handleListModelsClick = (event, index) => {
    
    setmodelList(Array.from(event))
    
  };


  function RenderResult({ isLoading }) {
    if (isLoading === null) {
      return <ResultComparaisonOne hidden={true} />
    }
    else if (isLoading === true) {
      return (
        <>
        
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: '4%' }}>
          <ResultComparaisonOne hidden={true} />
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
          <CircularProgress />
        </Box>
        
        </>
        
      )
    }
    else {
      if (state === null) {
        return <ResultComparaisonOne hidden={false} scoreResult={scoreResult} isEmpty={true} chartData={null} chart={false} table={null} />
      }
      else {
        return <ResultComparaisonOne hidden={false} scoreResult={null} isEmpty={false} chartData={state} chart={true} chartmetrics={metrics} modelsLength={modelsLength} chartLength={chartLength} table={array_cell} csv={csv}/>
      }

    }

  }

  function Item(props) {
    const { sx, ...other } = props;
    return (
      <Box
        sx={{
          p: 1,
          bgcolor: 'transparent',
          color: (theme) => (theme.palette.mode === 'dark' ? 'grey.300' : 'grey.800'),
          fontSize: '0.875rem',
          ...sx,
        }}
        {...other}
      />
    );
  }

  return (
    <>

      <Suspense fallback={<div>Loading...</div>}>
        <Sidebar selectedIndex={3} descriptionList={descriptionList} accuracy={accuracy} rappel={rappel} precision={precision} f1_score={f1_score} performances={performances} />
      </Suspense>
     
         
      <div id="firstSection">
      

        <div style={{ marginTop: '1%' }} >
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'row',
              p: 1,
              justifyContent: 'space-between',
              position: 'relative',
              width: '75%',
              height: '60px',
              margin: '1px 133px'
            }}
          >
            <Item sx={{ backgroundColor: 'none', height: '50px', width: '100%' }}>
              <Typography variant="h5" sx={{ fontFamily: 'Poppins', fontWeight: 500, color: '#5885FB' }}><LooksOneIcon sx={{ margin: '0 18px', height: '6%', width: '6%', color: "#ffab00" }} />Insertion des données</Typography>
            </Item>
            <Item sx={{ Color: '#0288d1', marginRight: '10%' }}>
            <div >
              <select  value={selectedOption} onChange={handleTestChange} id="shuffle" >
              <option value="">-- Select Test Option--</option>
               <option value="option1">Shuffle Test</option>
                <option value="option2">Suppression</option>
              </select>
            </div>
            </Item>
          </Box>
        </div>
      </div>
      <div className='form'>
        <Form onSubmit={handleSubmit} enctype="multipart/form-data">
          <div className='input_text'>
            <textarea
              id='CheckIt'
              className='_textarea'
              required
              type='text'
              placeholder="Insérez un texte"
              value={text}
              onChange={(e) => setText(e.target.value)
              }
              
            />
          </div>
          <br />
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'row-reverse',
              p: 1,
              justifyContent: 'flex-start',
              position: 'absolute',
              width: '63%',
              height: '80px',
              marginTop: '-1%',
              textAlign: 'center'
            }}
          >
            <Item >
              {/*<div className="file-inputs">
              <Button type="button" id='import_btn' onClick={handleImport}> {fileName ?? "Importer un fichier"} </Button>               
               <input type="file" ref={hiddenFileInput} onChange={handleFileInputChange} style={{ display: 'none' }} multiple />
              </div>*/}
             <div className="file-inputs">
                <Button type="button" id='import_btn' onClick={handleClickOpenDialog}> {fileName ?? "Importer un fichier"} </Button>
              </div>

            </Item>
            <Item><Typography sx={{ fontFamily: 'Poppins', fontSize: '16px', padding: '50% 0px' }}>Ou</Typography></Item>
          </Box>

          <div id="secondSection">
            <div>
              <Box
                sx={{
                  display: 'flex',
                  flexDirection: 'row',
                  justifyContent: 'space-between',
                  position: 'relative',
                  width: '75%',
                  height: '60px',
                  margin: '10px 133px'
                }}
              >
                <Item sx={{ backgroundColor: 'none', height: '50px', width: '100%' }}>
                  <Typography variant="h5" sx={{ fontFamily: 'Poppins', fontWeight: 500, color: '#5885FB' }}><LooksTwoIcon sx={{ margin: '0 18px', height: '5%', width: '5%', color: "#ffab00" }} />Sélection du modèle</Typography>
                </Item>
              </Box>
            </div>
          </div>
           
           
          <div className='eval_anal'>
      
 
            <div id='analyser_btn'>
            <Multiselect
        isObject={false}
        onRemove={(event) => {
          handleListModelsClick(event)
        }}
        onSelect={(event) => {
          handleListModelsClick(event)
          
        }}
        options={modelNames}
        selectedValues={["sent_avg"]}

        showCheckbox
      />
            </div>
            <Button type="submit" id='eval_btn'>Évaluer</Button>
          </div>
          <RenderResult isLoading={isLoading} />
        </Form>
      </div>


      <Dialog  aria-labelledby="customized-dialog-title" open={openDialog} fullWidth='true' maxWidth='sm' PaperProps={{
        style: {
       
        },


      }} >
     
       
       <div style={{margin:"10px  0px   10px 20%", padding:"10px", fontFamily: 'Poppins'}}>
                        <h4 >Selectionnez une option</h4>
       </div>
        <div style={{margin:"10px 25% 10px"}}>
        <p>Importer un csv</p>
        <Button onClick={handleImportcsv}  variant="outlined" style={{ width: "250px",
        height: "50px" ,padding: "6px 10px"}} startIcon={<FileUploadOutlinedIcon/>}>Importer un fichier</Button>
        <input type="file" ref={hiddenFileInput} onChange={handleChange} style={{ display: 'none' }} />
  
        <br></br><br></br>
        <p>Importer des fichiers textes</p>
        <Button onClick={handleImporttxt} variant="outlined" style={{ width: "250px",
        height: "50px" , marginBottom:"20px",padding: "6px 10px"}} startIcon={<FileUploadOutlinedIcon/>}>Importer un fichier</Button>
        <input type="file" ref={hiddenFile} onChange={handleFileInputChange} style={{ display: 'none' }} multiple />

         </div>
       
         
        </Dialog>
      
    </>

  )
}

export default Comparaison