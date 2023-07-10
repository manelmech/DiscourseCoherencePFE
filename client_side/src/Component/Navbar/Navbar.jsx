import React from 'react';
import './Navbar.css'
import logo from '../../assets/logo.png'
import { LanguageContext } from '../../context/LangageContext.js';
import { useContext } from 'react';







const Menu = () => {
    const { language, changeLanguage } = useContext(LanguageContext);
return(
    <>
        <div className='menu_div'> <p className='link'> <a href='accueil'> {language=="fr"? "Accueil" : "Home"}
 </a></p> </div>
        <div className='menu_div'> <p> <a href='apropos'> {language=="fr"? "A propos" : "About"}
 </a></p> </div>
        <div className='menu_div'> <p> <a href='connexion' id="connexion"> {language=="fr"? "Connexion" : "Log in"}
</a></p> </div>
    </>
)
}
const Navbar = () => {
    const { language, changeLanguage } = useContext(LanguageContext);
    const handleLanguageChange = (event) => {
        const newLanguage = event.target.value;
         changeLanguage(newLanguage);
        };
    return (
        
        <div id='navbar' >
            <div className='navbar_links' >
                < div className='navbar_logo' >
                    <img src={logo}
                        alt="logo" />
                    Coherencia
                </div>
                <div className='navbar_menu' >
                    <Menu />
                    
                    <select value={language} onChange={handleLanguageChange}>
                        <option value="en">En</option>
                        <option value="fr">Fr</option>
                
                    </select>
                </div>
            </div>
        </div>
    )
}

export default Navbar