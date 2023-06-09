import './App.css';
import {Accueil, Apropos, Navbar , Comparaison} from './Component';
import Login from './Component/Login/Login'
import ListeModele from './Component/GestionModeles/ListeModeles/ListeModele'
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { LanguageProvider } from './context/LangageContext';


// import { RequireToken } from "./Auth";

const App = () => {
  return (
    <LanguageProvider>
    <>
    <Navbar/>
    <Router>
        <Routes>
            <Route path='/connexion' element={<Login/>}/>

            <Route path='/' element={<Accueil/>}/>
            
            <Route path='/accueil' element={<Accueil/>}/>

            <Route path='/apropos' element={<Apropos/>}/>


            <Route path='/gestionmodeles' element={<ListeModele/>}/>
        </Routes>
    </Router>  
    </>    
    </LanguageProvider>

  );
}

export default App;
