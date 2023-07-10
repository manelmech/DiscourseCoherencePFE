import React from 'react'
import Card from '@mui/material/Card';
import Box from '@mui/material/Box';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Brightness1RoundedIcon from '@mui/icons-material/Brightness1Rounded';
import Button from '@mui/material/Button';
import { BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './ResultComparaisonOne.css';
import { styled } from '@mui/material/styles';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell, { tableCellClasses } from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import RemoveCircleOutlineIcon from '@mui/icons-material/RemoveCircleOutline';
import Looks3Icon from '@mui/icons-material/Looks3';
import { LanguageContext } from '../../context/LangageContext.js';
import { useContext } from 'react';
const ResultComparaisonOne = ({ hidden, scoreResult, isEmpty, chartData, chart, chartLength, table ,modelsLength ,csv, chartmetrics }) => {
  const [displayTable , setDisplay] = React.useState(false);
  const [buttonTexte , setButtonTexte] = React.useState("Afficher Plus");
  const [icon, setIcon] = React.useState(<AddCircleOutlineIcon/>)
  const { language, changeLanguage } = useContext(LanguageContext);

  const handleDisplayTable = (e) => {
    e.preventDefault();
    if(!displayTable){
      setDisplay(true)
      setButtonTexte("Réduire le tableau")
      setIcon(<RemoveCircleOut
      lineIcon/>)
    }
    else{
      setDisplay(false)
      setButtonTexte( language=="fr"? "Afficher Plus" : "Show more"
      )
      setIcon(<AddCircleOutlineIcon/>)
    }
    
  };

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

  function Score({ scoreResult }) {
    switch (scoreResult) {
      case null:
        return <></>
      default:
        return <BasicCard score={scoreResult} />
    }
  }

  function BasicCard({ score }) {
    let val
    if (score > 2) {
      val = <Typography variant="h6" component="div" color="#079615">
        Score de cohérence : {score}
      </Typography>
    }
    else if (score > 1 && score <= 2) {
      val = <Typography variant="h6" component="div" color="#FF9A02">
        Score de cohérence : {score}
      </Typography>
    }
    else if (score >= 0 && score <= 1) {
      val = <Typography variant="h6" component="div" color="#E33A3A">
        Score de cohérence : {score}
      </Typography>
    }
    else {
      val = <Typography variant="h6" component="div">
        Score de coherence : {score}
      </Typography>
    }
    return (
      <div className='result'>
        <Card sx={{ minWidth: 275, border: 1 }}>
          <CardContent>
            <Typography variant="h6" component="div">
              {val}
            </Typography>
          </CardContent>
        </Card>
      </div>
    );
  }

  function RenderChart({ isEmpty, chartData, chartLength, csv }) {
    if (isEmpty === true) {
      return <></>
    }
    else {
      var indents = [];
      for (var i = 0; i < chartLength; i++) {
        indents.push(<Bar dataKey={"Text"+i} fill="#ffab00" />);
      }
      return (
        <>

          <div style={{ display: 'block' , marginLeft: '50px',  marginTop: '90px' }}>
            <div style={{ display: 'flex', justifyContent: 'center' }}>
              <BarChart
                width={500}
                height={300}
                data={chartData}
                margin={{
                  top: 0,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey={"Low"} fill="#F31212" />
                <Bar dataKey={"Medium"} fill="#ffab00" />
                <Bar dataKey={"High"} fill="#4AC821" />

   
              </BarChart>
              {csv?
              <BarChart
                width={500}
                height={300}
                data={chartmetrics}
                margin={{
                  top: 0,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey={"Accuracy"} fill="#FF006C" />
                <Bar dataKey={"precision"} fill="#ffab00" />
                <Bar dataKey={"Recall"} fill="#269FFF" />
                <Bar dataKey={"F1_score"} fill="#BA27FF"  />


                
              </BarChart>
              :null}
            </div>
            <div>
            <div style={{ display: 'block' , marginLeft: '50px',  marginTop: '10px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-around'}}>
                <Typography variant='body2' sx={{ fontSize: 18, fontFamily: 'Didact Gothic', marginLeft: '-100px'  }}  color="#000" gutterBottom>
                  {language=="fr"? "Nombre de document par chaque classe" : "Number of documents per each class"}
                </Typography>
                <Typography variant='body2' sx={{ fontSize: 18, fontFamily: 'Didact Gothic' }} color="#000" gutterBottom>
                  {language=="fr"? " Métriques d'évaluation" : "Evaluation Metrics"}

                </Typography>
               
               
                </div>
              </div>
            </div>
          </div>
          
          
          <Button variant="outlined"  endIcon={icon}  onClick={handleDisplayTable}>{buttonTexte}</Button>
          <div style={{ display: 'block' , marginLeft: '100px'  }}>

          <RenderTable rows={table} displayTable = {displayTable} chartLength= {chartLength} modelsLength={modelsLength} csv={csv} > </RenderTable>
           </div>
        </>

      )
    }
  }



  function Render({ chart, csv }) {
    if (chart === true) {
      return (
        <>
          <div>
            <RenderChart isEmpty={isEmpty} chartData={chartData} chartLength={chartLength} csv={csv}/>
          </div>

        </>
      )
    }
    else {
      return (
        <>
          <Score scoreResult={scoreResult} />
          <div className='cards-key'>
            <div className='card1'>
              <div><Brightness1RoundedIcon sx={{ color: "#079615" }} /></div>
              <p id='scoreCard'> 3 (élevé)</p>
            </div>
            <div className='card1'>
              <div><Brightness1RoundedIcon sx={{ color: "#FF9A02" }} /></div>
              <p id='scoreCard'> 2 (moyen)</p>
            </div>
            <div className='card1'>
              <div><Brightness1RoundedIcon sx={{ color: "#E33A3A" }} /></div>
              <p id='scoreCard'> 1 (bas)</p>
            </div>
          </div>
        </>
      )
    }
  }


  const StyledTableCell = styled(TableCell)(({ theme }) => ({
    [`&.${tableCellClasses.head}`]: {
      backgroundColor: "#0288d1",
      color: theme.palette.common.white,
    },
    [`&.${tableCellClasses.body}`]: {
      fontSize: 14,
    },
  }));

  const StyledTableRow = styled(TableRow)(({ theme }) => ({
    '&:nth-of-type(odd)': {
      backgroundColor: theme.palette.action.hover,
    },
    // hide last border
    '&:last-child td, &:last-child th': {
      border: 0,
    },
  }));

  function RenderTable({ rows, displayTable , modelsLength , csv }) {
    if (displayTable ==false){
      return
    }else{
    return (
      <div>
      <Table sx={{ minWidth: 70, maxWidth: 1000 , height: "200px", overflow:"scroll"}} aria-label="customized table">
        <TableHead>
          <TableRow>
            {(csv)?
            <StyledTableCell sx={{fontFamily : 'Didact Gothic', fontWeight :"bold"}} >{language=="fr"? "ID du document " : "Document Id"}
            </StyledTableCell>
            :null}
            <StyledTableCell align="center" sx={{fontFamily : 'Didact Gothic', fontWeight :"bold"}}>{language=="fr"? "Texte" : "Text"}</StyledTableCell>
            <StyledTableCell align="left" sx={{fontFamily : 'Didact Gothic', fontWeight :"bold" }} >{language=="fr"? "Nom du modéle" : "Model Name"}</StyledTableCell>

            <StyledTableCell align="left" sx={{fontFamily : 'Didact Gothic', fontWeight :"bold"}}>{language=="fr"? "Score prédit" : "Predicted score"}</StyledTableCell>
            {(csv)?
            <StyledTableCell align="left" sx={{fontFamily : 'Didact Gothic', fontWeight :"bold"}}>{language=="fr"? "Score original" : "Original score"}</StyledTableCell>
            :null}
          </TableRow>
        </TableHead>
        <TableBody>

          {rows.map((row,i) => (
              
            <StyledTableRow >
              {(csv && i%modelsLength==0)?
              
              <StyledTableCell rowspan={modelsLength} component="th" scope="row">
                {row.text_id}
              </StyledTableCell>:null}   
              {(i%modelsLength==0)?   
              <StyledTableCell rowspan={modelsLength} align="justify" sx={{width: "400px", height : "200px", fontFamily : 'Didact Gothic'}}><div style={{width: "400px", height: "200px" , overflow: "auto"}}>{row.text}</div></StyledTableCell>
              :null}
              <StyledTableCell align="center" sx={{width: "100px", height : "200px", fontFamily : 'Didact Gothic'}}><div style={{width: "100px", height: "200px" , overflow: "auto"}}>{row.modelname}</div></StyledTableCell>

              <StyledTableCell align="center" sx={{ fontWeight: "bold" , fontFamily : 'Didact Gothic'}}>{row.predicted_score}</StyledTableCell>
              {(csv)?
              <StyledTableCell align="center" sx={{fontFamily : 'Didact Gothic'}}>{row.original_score}</StyledTableCell>
              :null}
            </StyledTableRow>
          ))}
        </TableBody>
      </Table>
    </div>

    );
            }

  }
  
  return (
    <>
     
      <div id='evalSection' hidden={hidden}>
        <Render chart={chart} csv={csv} />
      </div>
    </>
    
  )
}

export default ResultComparaisonOne