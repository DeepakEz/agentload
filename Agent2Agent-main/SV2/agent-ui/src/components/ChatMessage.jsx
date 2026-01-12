import React from "react";

export default function ChatMessage({ text, type, theme }) {
  const baseStyle = {
    padding: "10px 15px",
    borderRadius: "12px",
    maxWidth: "90%",
    wordWrap: "break-word",
    fontSize: "14px",
    whiteSpace: "pre-wrap"
  };

  const userStyle = {
    ...baseStyle,
    backgroundColor: theme.chatItemBg,
    color: theme.text,
    alignSelf: "flex-end"
  };

  const agentStyle = {
    ...baseStyle,
    backgroundColor: theme.chatItemBg,
    color: theme.text,
    alignSelf: "flex-start"
  };

  const insightStyle = {
    ...baseStyle,
    backgroundColor: "#1a472a",  // Dark green for insights
    color: "#90EE90",  // Light green text
    alignSelf: "center",
    border: "2px solid #2e7d32",
    fontStyle: "italic",
    maxWidth: "95%",
    textAlign: "center",
    animation: "fadeIn 0.5s"
  };

  const getStyle = () => {
    if (type === "user") return userStyle;
    if (type === "insight") return insightStyle;
    return agentStyle;
  };

  return (
    <>
      <div style={getStyle()}>{text}</div>
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </>
  );
}
