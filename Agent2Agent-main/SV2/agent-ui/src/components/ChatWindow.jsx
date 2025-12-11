import React, { useState, useRef } from "react";
import ChatMessage from "./ChatMessage";
import { FaRocket, FaMicrophone } from "react-icons/fa";

export default function ChatWindow({ theme }) {
  const [messages, setMessages] = useState([
    { text: "Hello! I'm your AI agent.", type: "agent" }
  ]);
  const [input, setInput] = useState("");
  const [loadingReply, setLoadingReply] = useState(false);
  const messagesEndRef = useRef(null);

  async function sendUserMessage(userMessage) {
    const res = await fetch("http://localhost:8000/process_message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_prompt: userMessage })
    });
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    return data.response;
  }

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg = input;
    setMessages(prev => [...prev, { text: userMsg, type: "user" }]);
    setInput("");
    setLoadingReply(true);

    try {
      const reply = await sendUserMessage(userMsg);
      setMessages(prev => [...prev, { text: reply, type: "agent" }]);
    } catch (err) {
      setMessages(prev => [...prev, { text: "Error contacting agent. Try again.", type: "agent" }]);
      console.error(err);
    } finally {
      setLoadingReply(false);
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
      flex: "0 0 60px",
      backgroundColor: theme.sidebarBg,
      display: "flex",
      alignItems: "center",
      padding: "0 20px",
      boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
      fontWeight: "bold",
      fontSize: "18px",
      color: theme.text
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
      <div style={styles.chatHeader}>Chat with Agent</div>

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
      <style jsx>{`
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
