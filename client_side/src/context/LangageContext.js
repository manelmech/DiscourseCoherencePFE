// LanguageContext.js
import React, { createContext, useState, useEffect } from 'react';


const LanguageContext = createContext();


const LanguageProvider = ({ children }) => {


  const storedLanguage = localStorage.getItem('language'); // Get the stored language from localStorage
  const [language, setLanguage] = useState(storedLanguage || 'en'); // Use the stored language or default to 'en'
 
  const changeLanguage = (newLanguage) => {
    setLanguage(newLanguage);
  };
  useEffect(() => {
    localStorage.setItem('language', language); // Save the selected language to localStorage
  }, [language]);
  return (
    <LanguageContext.Provider value={{ language, changeLanguage }}>
      {children}
    </LanguageContext.Provider>
  );
};


export { LanguageContext, LanguageProvider };



