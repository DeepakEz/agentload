// Sidebar.jsx
import React, { useState, useEffect } from "react";

const themes = {
  dark: {
    sidebarBg: "#000000",
    text: "#ffffff",
    buttonBg: "#1a1a1a",
    buttonHover: "#333333",
    chatItemBg: "#1a1a1a",
    chatItemHover: "#333333",
    bottomButtonBg: "#1a1a1a",
    bottomButtonHover: "#333333",
    toggleBg: "#ffffff",
    toggleText: "#000000",
    chatBg: "#121212",
  },
  light: {
    sidebarBg: "#ffffff",
    text: "#000000",
    buttonBg: "#e0e0e0",
    buttonHover: "#cccccc",
    chatItemBg: "#f5f5f5",
    chatItemHover: "#e0e0e0",
    bottomButtonBg: "#e0e0e0",
    bottomButtonHover: "#cccccc",
    toggleBg: "#000000",
    toggleText: "#ffffff",
    chatBg: "#f7f7f8",
  }
};

const styles = {
  container: (theme, isCollapsed) => ({
    width: isCollapsed ? "70px" : "280px",
    height: "100vh",
    backgroundColor: theme.sidebarBg,
    color: theme.text,
    display: "flex",
    flexDirection: "column",
    padding: isCollapsed ? "10px 5px" : "20px",
    boxSizing: "border-box",
    transition: "width 0.3s",
    position: "relative",
    fontFamily: "'Courier New', Courier, monospace"
  }),
  topSection: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    marginBottom: "20px"
  },
  title: {
    fontSize: "18px",
    fontWeight: "bold",
    marginBottom: "15px",
    textAlign: "center"
  },
  newChatButton: (theme) => ({
    width: "100%",
    padding: "10px",
    borderRadius: "8px",
    border: "none",
    cursor: "pointer",
    fontWeight: "bold",
    backgroundColor: theme.buttonBg,
    color: theme.text,
    marginBottom: "20px",
    transition: "background-color 0.2s"
  }),
  chatHistoryTitle: {
    fontSize: "14px",
    fontWeight: "bold",
    marginBottom: "10px",
    paddingLeft: "5px"
  },
  chatListBlock: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    overflowY: "auto",
    paddingRight: "5px"
  },
  chatItem: (theme) => ({
    padding: "8px 12px",
    borderRadius: "6px",
    cursor: "pointer",
    backgroundColor: theme.chatItemBg,
    transition: "background-color 0.2s",
    fontSize: "14px",
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis"
  }),
  bottomSection: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "10px",
    marginTop: "auto"
  },
  bottomButton: (theme) => ({
    width: "90%",
    padding: "10px",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    color: theme.text,
    textAlign: "center",
    transition: "background-color 0.2s",
    backgroundColor: theme.bottomButtonBg
  }),
  agentsList: {
    marginTop: "5px",
    display: "flex",
    flexDirection: "column",
    gap: "5px",
    paddingLeft: "10px"
  },
  agentsItem: (theme) => ({
    fontSize: "13px",
    cursor: "pointer",
    padding: "5px 8px",
    borderRadius: "5px",
    backgroundColor: theme.chatItemBg,
    transition: "background-color 0.2s"
  }),
  toggleButton: (theme) => ({
    position: "absolute",
    top: "50%",
    left: "100%",
    transform: "translate(-50%, -50%)",
    backgroundColor: theme.toggleBg,
    border: "none",
    borderRadius: "50%",
    width: "35px",
    height: "35px",
    cursor: "pointer",
    color: theme.toggleText,
    fontWeight: "bold",
    boxShadow: "0 2px 6px rgba(0,0,0,0.3)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center"
  }),
  version: { fontSize: "12px" }
};

export default function Sidebar({ theme, toggleTheme }) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [showAgents, setShowAgents] = useState(false);

  const previousChats = ["Chat with AI 1", "Chat with AI 2", "Chat with AI 3"];
const [agents, setAgents] = useState([]);
useEffect(() => {
  fetch("/agents.json")
    .then(res => {
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return res.json();
    })
    .then(data => {
      // Convert object keys into array of agent objects
      const agentsArray = Object.keys(data).map(key => ({
        name: key,
        system_instruction: data[key].system_instruction
      }));
      setAgents(agentsArray);
    })
    .catch(err => console.error("Error loading agents:", err));
}, []);


  return (
    <div style={{ position: "relative" }}>
      <div style={styles.container(theme, isCollapsed)}>
        {/* Top Section */}
        {!isCollapsed && (
          <div style={styles.topSection}>
            <div style={styles.title}>AGENT BUILDER</div>
            <button
              style={styles.newChatButton(theme)}
              onMouseEnter={e => (e.currentTarget.style.backgroundColor = theme.buttonHover)}
              onMouseLeave={e => (e.currentTarget.style.backgroundColor = theme.buttonBg)}
            >
              + New Chat
            </button>
          </div>
        )}

        {/* Chat History */}
        {!isCollapsed && (
          <>
            <div style={styles.chatHistoryTitle}>Chat History</div>
            <div style={styles.chatListBlock}>
              {previousChats.map((chat, idx) => (
                <div
                  key={idx}
                  style={styles.chatItem(theme)}
                  onMouseEnter={e => (e.currentTarget.style.backgroundColor = theme.chatItemHover)}
                  onMouseLeave={e => (e.currentTarget.style.backgroundColor = theme.chatItemBg)}
                >
                  {chat}
                </div>
              ))}
            </div>
          </>
        )}

        {/* Bottom Section */}
        {!isCollapsed && (
          <div style={styles.bottomSection}>
            {/* Dark/Light Mode */}
            <button
              style={styles.bottomButton(theme)}
              onClick={toggleTheme}
              onMouseEnter={e => (e.currentTarget.style.backgroundColor = theme.buttonHover)}
              onMouseLeave={e => (e.currentTarget.style.backgroundColor = theme.bottomButtonBg)}
            >
              {theme === themes.dark ? "🌙 Dark" : "☀️ Light"}
            </button>

            <button
              style={styles.bottomButton(theme)}
              onClick={() => setShowAgents(!showAgents)}
              onMouseEnter={e => (e.currentTarget.style.backgroundColor = theme.bottomButtonHover)}
              onMouseLeave={e => (e.currentTarget.style.backgroundColor = theme.bottomButtonBg)}
            >
              Agents
            </button>
            {showAgents && (
  <div style={styles.agentsList}>
    {agents.map((agent, idx) => (
      <div
        key={idx}
        style={styles.agentsItem(theme)}
        onMouseEnter={e => (e.currentTarget.style.backgroundColor = theme.chatItemHover)}
        onMouseLeave={e => (e.currentTarget.style.backgroundColor = theme.chatItemBg)}
      >
        {agent.name}  {/* you can show system_instruction if needed */}
      </div>
    ))}
  </div>
)}



            <button
              style={styles.bottomButton(theme)}
              onMouseEnter={e => (e.currentTarget.style.backgroundColor = theme.bottomButtonHover)}
              onMouseLeave={e => (e.currentTarget.style.backgroundColor = theme.bottomButtonBg)}
            >
              Settings
            </button>
            <div style={{...styles.version, color: theme.text}}>v1.0</div>
          </div>
        )}
      </div>

      {/* Toggle Button */}
      <button
        style={styles.toggleButton(theme)}
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        {isCollapsed ? "🔒" : "🔓"}
      </button>
    </div>
  );
}
