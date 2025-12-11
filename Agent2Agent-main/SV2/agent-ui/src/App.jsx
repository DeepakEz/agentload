import React, { useState } from "react";  // <-- import useState
import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";

const styles = {
  app: {
    display: "flex",
    height: "100vh",
    width: "100vw",      // full viewport width
    fontFamily: "Arial, sans-serif",
    overflow: "hidden"   // prevent unwanted scrollbars
  }
};

const themes = {
  dark: {
    sidebarBg: "#000000",        // solid black sidebar
    text: "#ffffff",             // bright white text
    buttonBg: "#111111",         // very dark gray buttons
    buttonHover: "#898989ff",      // subtle gray for hover
    chatItemBg: "#282727ff",       // dark gray chat history blocks
    chatItemHover: "#898989ff",    // hover slightly lighter
    bottomButtonBg: "#111111",   // bottom buttons dark gray
    bottomButtonHover: "#898989ff",// hover for bottom buttons
    toggleBg: "#ffffff",          // toggle button white
    toggleText: "#000000",        // toggle icon black
    chatBg: "#0f0f0f"  ,
    scrollTrack: "#121212",
  scrollThumb: "#333333"           // chat window black with slight gray
  },
  light: {
    sidebarBg: "#f5f5f5",
    text: "#0d0d0d",
    buttonBg: "#e0e0e0",
    buttonHover: "#cccccc",
    chatItemBg: "#e0e0e0",
    chatItemHover: "#cccccc",
    bottomButtonBg: "#e0e0e0",
    bottomButtonHover: "#cccccc",
    toggleBg: "#0d0d0d",
    toggleText: "#f5f5f5",
    chatBg: "#f7f7f8",
    scrollTrack: "#f7f7f8",
  scrollThumb: "#cccccc"
  }
};




function App() {
  const [themeName, setThemeName] = useState("dark"); // <-- fixed
  const toggleTheme = () => setThemeName(prev => (prev === "dark" ? "light" : "dark"));

  const theme = themes[themeName];

  return (
    <div style={styles.app}>
      <Sidebar theme={theme} toggleTheme={toggleTheme} />
      <ChatWindow theme={theme} />
    </div>
  );
}

export default App;
