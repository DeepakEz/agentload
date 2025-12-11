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
    backgroundColor: theme.chatItemBg,   // user message background
    color: theme.text,                 // user message text
    alignSelf: "flex-end"
  };

  const agentStyle = {
    ...baseStyle,
    backgroundColor: theme.chatItemBg, // agent message background
    color: theme.text,                 // agent message text
    alignSelf: "flex-start"
  };

  return <div style={type === "user" ? userStyle : agentStyle}>{text}</div>;
}
