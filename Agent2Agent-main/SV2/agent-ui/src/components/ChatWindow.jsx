import React, { useState, useRef, useEffect } from "react";
import ChatMessage from "./ChatMessage";
import { FaRocket, FaMicrophone } from "react-icons/fa";

export default function ChatWindow({ theme }) {
  const [messages, setMessages] = useState([
    { text: "Hello! I'm your AI agent. Select an agent from the dropdown or chat with the default assistant.", type: "agent" }
  ]);
  const [input, setInput] = useState("");
  const [loadingReply, setLoadingReply] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [availableAgents, setAvailableAgents] = useState([]);
  const [agentWisdom, setAgentWisdom] = useState(null);
  const [tokensUsed, setTokensUsed] = useState(0);
  const messagesEndRef = useRef(null);

  // Fetch available agents on component mount
  useEffect(() => {
    async function fetchAgents() {
      try {
        const res = await fetch("http://localhost:8000/agents");
        if (res.ok) {
          const data = await res.json();
          setAvailableAgents(data.agents || []);
        }
      } catch (err) {
        console.error("Failed to fetch agents:", err);
      }
    }
    fetchAgents();
  }, []);

  // Poll for insights when agent is selected
  useEffect(() => {
    if (!selectedAgent) return;

    async function checkInsights() {
      try {
        const res = await fetch(`http://localhost:8000/agents/${selectedAgent}/insights`);
        if (res.ok) {
          const data = await res.json();
          if (data.insights && data.insights.length > 0) {
            // Add insights as agent-initiated messages
            for (const insight of data.insights) {
              setMessages(prev => [...prev, {
                text: `💡 ${insight.text}`,
                type: "insight",
                insightId: insight.id
              }]);

              // Mark as delivered
              await fetch(`http://localhost:8000/agents/${selectedAgent}/insights/${insight.id}/delivered`, {
                method: "POST"
              });
            }
          }
        }

        // Fetch agent wisdom
        const wisdomRes = await fetch(`http://localhost:8000/agents/${selectedAgent}/wisdom`);
        if (wisdomRes.ok) {
          const wisdomData = await wisdomRes.json();
          setAgentWisdom(wisdomData);
        }
      } catch (err) {
        console.error("Failed to check insights:", err);
      }
    }

    checkInsights();
    const interval = setInterval(checkInsights, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, [selectedAgent]);

  async function sendUserMessage(userMessage, agentId, conversationHistory) {
    const payload = {
      user_prompt: userMessage,
      agent_id: agentId || null,
      conversation_history: conversationHistory || null
    };

    const res = await fetch("http://localhost:8000/process_message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();

    // Update tokens used
    if (data.tokens_used) {
      setTokensUsed(data.tokens_used);
    }

    return data;
  }

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg = input;
    const newMessages = [...messages, { text: userMsg, type: "user" }];
    setMessages(newMessages);
    setInput("");
    setLoadingReply(true);

    try {
      // Pass conversation history (last 10 messages) and selected agent
      const data = await sendUserMessage(userMsg, selectedAgent, messages.slice(-10));
      setMessages(prev => [...prev, { text: data.response, type: "agent" }]);

      // Show pending insights if any
      if (data.pending_insights && data.pending_insights.length > 0) {
        for (const insight of data.pending_insights) {
          setMessages(prev => [...prev, {
            text: `💡 ${insight.text}`,
            type: "insight",
            insightId: insight.id
          }]);
        }
      }
    } catch (err) {
      setMessages(prev => [...prev, { text: "Error contacting agent. Try again.", type: "agent" }]);
      console.error(err);
    } finally {
      setLoadingReply(false);
    }
  };

  const handleAgentChange = (e) => {
    const agentId = e.target.value || null;
    setSelectedAgent(agentId);

    // Add a system message when agent changes
    if (agentId) {
      const agent = availableAgents.find(a => a.id === agentId);
      setMessages(prev => [...prev, {
        text: `Switched to ${agent.name}. I'm now following this instruction: "${agent.system_instruction}"`,
        type: "agent"
      }]);
    } else {
      setMessages(prev => [...prev, {
        text: "Switched to default assistant (no specific agent).",
        type: "agent"
      }]);
    }
  };

  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // -------- Styles --------
  const styles = {
    iconButton: {
      marginLeft: "10px",
      padding: "10px",
      border: "none",
      borderRadius: "8px",
      backgroundColor: theme.buttonBg,
      cursor: loadingReply ? "not-allowed" : "pointer",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      opacity: loadingReply ? 0.6 : 1
    },
    chatContainer: { flex: 1, display: "flex", flexDirection: "column", backgroundColor: theme.chatBg },
    chatHeader: {
      flex: "0 0 70px",
      backgroundColor: theme.sidebarBg,
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: "0 20px",
      boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
      fontWeight: "bold",
      fontSize: "18px",
      color: theme.text
    },
    agentSelector: {
      padding: "8px 12px",
      borderRadius: "8px",
      border: `1px solid ${theme.buttonHover}`,
      backgroundColor: theme.chatBg,
      color: theme.text,
      fontSize: "14px",
      cursor: "pointer",
      minWidth: "200px"
    },
    wisdomBadge: {
      fontSize: "11px",
      padding: "4px 8px",
      borderRadius: "12px",
      backgroundColor: theme.buttonBg,
      marginLeft: "10px",
      display: "inline-block"
    },
    tokenCounter: {
      fontSize: "11px",
      opacity: 0.7,
      marginTop: "2px"
    },
    headerInfo: {
      display: "flex",
      flexDirection: "column",
      alignItems: "flex-start"
    },
    messagesArea: {
      flex: 1,
      padding: "20px",
      overflowY: "auto",
      display: "flex",
      flexDirection: "column",
      gap: "10px",
      scrollbarWidth: "thin",              // Firefox
      scrollbarColor: `${theme.scrollThumb} ${theme.scrollTrack}` // Firefox
    },
    inputArea: { flex: "0 0 60px", display: "flex", padding: "10px 20px", backgroundColor: theme.sidebarBg, boxShadow: "0 -2px 4px rgba(0,0,0,0.1)" },
    input: { flex: 1, padding: "10px", borderRadius: "8px", border: `1px solid ${theme.buttonHover}`, fontSize: "16px", backgroundColor: theme.chatBg, color: theme.text }
  };

  return (
    <div style={styles.chatContainer}>
      <div style={styles.chatHeader}>
        <div style={styles.headerInfo}>
          <div>
            <span>
              {selectedAgent
                ? `Chatting with: ${availableAgents.find(a => a.id === selectedAgent)?.name}`
                : "Chat with Agent"}
            </span>
            {agentWisdom && (
              <span style={styles.wisdomBadge}>
                📚 {agentWisdom.conversation_count} convos
                {agentWisdom.expertise_areas && agentWisdom.expertise_areas.length > 0 &&
                  ` | ${agentWisdom.expertise_areas.slice(0, 3).join(", ")}`
                }
              </span>
            )}
          </div>
          {tokensUsed > 0 && (
            <div style={styles.tokenCounter}>
              Context: {tokensUsed} / 2048 tokens {tokensUsed > 1800 && "⚠️ Near limit"}
            </div>
          )}
        </div>
        <select
          style={styles.agentSelector}
          value={selectedAgent || ""}
          onChange={handleAgentChange}
        >
          <option value="">Default Assistant</option>
          {availableAgents.map(agent => (
            <option key={agent.id} value={agent.id}>
              {agent.name}
            </option>
          ))}
        </select>
      </div>

      <div
        style={styles.messagesArea}
        className="messages-area"
      >
        {messages.map((msg, idx) => (
          <ChatMessage key={idx} text={msg.text} type={msg.type} theme={theme} />
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div style={styles.inputArea}>
        <input
          style={styles.input}
          type="text"
          placeholder={loadingReply ? "Waiting for agent reply..." : "Type a message..."}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleSend()}
          disabled={loadingReply}
        />
        <button style={styles.iconButton} disabled={loadingReply}>
          <FaMicrophone size={22} color={theme.text} />
        </button>
        <button style={styles.iconButton} onClick={handleSend} disabled={loadingReply}>
          {loadingReply ? "Loading..." : <FaRocket size={22} color={theme.text} />}
        </button>
      </div>

      {/* Scrollbar CSS for Webkit browsers */}
      <style>{`
        .messages-area::-webkit-scrollbar {
          width: 8px;
        }
        .messages-area::-webkit-scrollbar-track {
          background: ${theme.scrollTrack};
        }
        .messages-area::-webkit-scrollbar-thumb {
          background-color: ${theme.scrollThumb};
          border-radius: 4px;
        }
      `}</style>
    </div>
  );
}
