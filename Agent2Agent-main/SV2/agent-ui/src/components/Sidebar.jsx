// Sidebar.jsx
import React, { useState, useEffect } from "react";
import ModelManager from "./ModelManager";

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
  const [showModelManager, setShowModelManager] = useState(false);
  const [showAgentCreator, setShowAgentCreator] = useState(false);

  const previousChats = ["Chat with AI 1", "Chat with AI 2", "Chat with AI 3"];
const [agents, setAgents] = useState([]);

const loadAgents = () => {
  fetch("http://localhost:8000/agents")
    .then(res => {
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return res.json();
    })
    .then(data => {
      setAgents(data.agents || []);
    })
    .catch(err => console.error("Error loading agents:", err));
};

useEffect(() => {
  loadAgents();
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
    <button
      style={{...styles.agentsItem(theme), fontWeight: "bold", textAlign: "center"}}
      onClick={() => setShowAgentCreator(true)}
      onMouseEnter={e => (e.currentTarget.style.backgroundColor = theme.chatItemHover)}
      onMouseLeave={e => (e.currentTarget.style.backgroundColor = theme.chatItemBg)}
    >
      + Add New Agent
    </button>
    {agents.map((agent, idx) => (
      <div
        key={idx}
        style={styles.agentsItem(theme)}
        onMouseEnter={e => (e.currentTarget.style.backgroundColor = theme.chatItemHover)}
        onMouseLeave={e => (e.currentTarget.style.backgroundColor = theme.chatItemBg)}
      >
        {agent.name}
      </div>
    ))}
  </div>
)}

            <button
              style={styles.bottomButton(theme)}
              onClick={() => setShowModelManager(true)}
              onMouseEnter={e => (e.currentTarget.style.backgroundColor = theme.bottomButtonHover)}
              onMouseLeave={e => (e.currentTarget.style.backgroundColor = theme.bottomButtonBg)}
            >
              🤖 Models
            </button>

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

      {/* Model Manager Modal */}
      {showModelManager && (
        <ModelManager
          theme={theme === themes.dark ? themes.dark : themes.light}
          onClose={() => setShowModelManager(false)}
        />
      )}

      {/* Agent Creator Modal */}
      {showAgentCreator && (
        <AgentCreator
          theme={theme === themes.dark ? themes.dark : themes.light}
          onClose={() => {
            setShowAgentCreator(false);
            loadAgents(); // Reload agents after creating
          }}
        />
      )}
    </div>
  );
}

// AgentCreator Modal Component
function AgentCreator({ theme, onClose }) {
  const [step, setStep] = useState(1); // 1 = describe, 2 = preview
  const [description, setDescription] = useState("");
  const [agentName, setAgentName] = useState("");
  const [systemInstruction, setSystemInstruction] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [generating, setGenerating] = useState(false);

  const handleGenerate = async () => {
    setError("");
    setGenerating(true);

    if (!description.trim()) {
      setError("Please describe what kind of agent you want");
      setGenerating(false);
      return;
    }

    try {
      const response = await fetch("http://localhost:8000/agents/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description })
      });

      const data = await response.json();

      if (data.success) {
        setAgentName(data.agent_name);
        setSystemInstruction(data.system_instruction);
        setStep(2); // Move to preview step
      } else {
        setError(data.error || "Failed to generate agent");
      }
    } catch (err) {
      setError("Network error: " + err.message);
    } finally {
      setGenerating(false);
    }
  };

  const handleCreate = async () => {
    setError("");

    try {
      const response = await fetch("http://localhost:8000/agents", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: agentName,
          system_instruction: systemInstruction
        })
      });

      const data = await response.json();

      if (data.success) {
        setSuccess(data.message);
        setTimeout(() => {
          onClose();
        }, 1500);
      } else {
        setError(data.error || "Failed to create agent");
      }
    } catch (err) {
      setError("Network error: " + err.message);
    }
  };

  return (
    <div style={{
      position: "fixed",
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: "rgba(0,0,0,0.7)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      zIndex: 1000
    }}>
      <div style={{
        backgroundColor: theme.sidebarBg,
        color: theme.text,
        padding: "30px",
        borderRadius: "10px",
        width: "500px",
        maxWidth: "90%"
      }}>
        <h2 style={{ marginTop: 0 }}>
          {step === 1 ? "Create New Agent" : "Review & Edit Agent"}
        </h2>

        {step === 1 ? (
          // Step 1: Describe what you want
          <>
            <div style={{ marginBottom: "20px" }}>
              <label style={{ display: "block", marginBottom: "8px", fontWeight: "bold" }}>
                What kind of agent do you want?
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Example: I want an agent that helps me write Python code and debug errors"
                rows={4}
                style={{
                  width: "100%",
                  padding: "10px",
                  borderRadius: "5px",
                  border: "1px solid #444",
                  backgroundColor: theme.chatItemBg,
                  color: theme.text,
                  fontSize: "14px",
                  resize: "vertical"
                }}
              />
            </div>

            {error && (
              <div style={{
                padding: "10px",
                marginBottom: "10px",
                backgroundColor: "#ff4444",
                color: "white",
                borderRadius: "5px"
              }}>
                {error}
              </div>
            )}

            <div style={{ display: "flex", gap: "10px" }}>
              <button
                onClick={handleGenerate}
                disabled={generating}
                style={{
                  flex: 1,
                  padding: "12px",
                  backgroundColor: generating ? "#666" : "#4CAF50",
                  color: "white",
                  border: "none",
                  borderRadius: "5px",
                  cursor: generating ? "not-allowed" : "pointer",
                  fontWeight: "bold"
                }}
              >
                {generating ? "Generating..." : "✨ Generate Agent"}
              </button>
              <button
                onClick={onClose}
                style={{
                  flex: 1,
                  padding: "12px",
                  backgroundColor: theme.buttonBg,
                  color: theme.text,
                  border: "none",
                  borderRadius: "5px",
                  cursor: "pointer"
                }}
              >
                Cancel
              </button>
            </div>
          </>
        ) : (
          // Step 2: Preview and edit
          <>
            <div style={{ marginBottom: "15px", padding: "10px", backgroundColor: theme.chatItemBg, borderRadius: "5px" }}>
              <small style={{ color: "#888" }}>You can edit the generated details below before creating</small>
            </div>

            <div style={{ marginBottom: "20px" }}>
              <label style={{ display: "block", marginBottom: "8px", fontWeight: "bold" }}>
                Agent Name
              </label>
              <input
                type="text"
                value={agentName}
                onChange={(e) => setAgentName(e.target.value)}
                style={{
                  width: "100%",
                  padding: "10px",
                  borderRadius: "5px",
                  border: "1px solid #444",
                  backgroundColor: theme.chatItemBg,
                  color: theme.text,
                  fontSize: "14px"
                }}
              />
            </div>

            <div style={{ marginBottom: "20px" }}>
              <label style={{ display: "block", marginBottom: "8px", fontWeight: "bold" }}>
                System Instruction
              </label>
              <textarea
                value={systemInstruction}
                onChange={(e) => setSystemInstruction(e.target.value)}
                rows={6}
                style={{
                  width: "100%",
                  padding: "10px",
                  borderRadius: "5px",
                  border: "1px solid #444",
                  backgroundColor: theme.chatItemBg,
                  color: theme.text,
                  fontSize: "14px",
                  resize: "vertical"
                }}
              />
            </div>

            {error && (
              <div style={{
                padding: "10px",
                marginBottom: "10px",
                backgroundColor: "#ff4444",
                color: "white",
                borderRadius: "5px"
              }}>
                {error}
              </div>
            )}

            {success && (
              <div style={{
                padding: "10px",
                marginBottom: "10px",
                backgroundColor: "#44ff44",
                color: "black",
                borderRadius: "5px"
              }}>
                {success}
              </div>
            )}

            <div style={{ display: "flex", gap: "10px" }}>
              <button
                onClick={() => setStep(1)}
                style={{
                  padding: "12px 20px",
                  backgroundColor: theme.buttonBg,
                  color: theme.text,
                  border: "none",
                  borderRadius: "5px",
                  cursor: "pointer"
                }}
              >
                ← Back
              </button>
              <button
                onClick={handleCreate}
                style={{
                  flex: 1,
                  padding: "12px",
                  backgroundColor: "#4CAF50",
                  color: "white",
                  border: "none",
                  borderRadius: "5px",
                  cursor: "pointer",
                  fontWeight: "bold"
                }}
              >
                Create Agent
              </button>
              <button
                onClick={onClose}
                style={{
                  padding: "12px 20px",
                  backgroundColor: theme.buttonBg,
                  color: theme.text,
                  border: "none",
                  borderRadius: "5px",
                  cursor: "pointer"
                }}
              >
                Cancel
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
