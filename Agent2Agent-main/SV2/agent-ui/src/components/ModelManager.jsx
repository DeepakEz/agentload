import React, { useState, useEffect } from "react";
import { FaDownload, FaCheck, FaTrash, FaSpinner, FaPlus } from "react-icons/fa";

export default function ModelManager({ theme, onClose }) {
  const [models, setModels] = useState([]);
  const [activeModelId, setActiveModelId] = useState(null);
  const [downloadProgress, setDownloadProgress] = useState({});
  const [customModelUrl, setCustomModelUrl] = useState("");
  const [showAddCustom, setShowAddCustom] = useState(false);
  const [loading, setLoading] = useState(false);

  // Fetch models on mount
  useEffect(() => {
    fetchModels();
    const interval = setInterval(fetchModels, 2000); // Poll every 2 seconds for progress
    return () => clearInterval(interval);
  }, []);

  async function fetchModels() {
    try {
      const res = await fetch("http://localhost:8000/models");
      if (res.ok) {
        const data = await res.json();
        setModels(data.models || []);
        setActiveModelId(data.active_model_id);

        // Fetch progress for downloading models
        for (const model of data.models) {
          if (!model.downloaded && !model.loaded) {
            fetchDownloadProgress(model.id);
          }
        }
      }
    } catch (err) {
      console.error("Failed to fetch models:", err);
    }
  }

  async function fetchDownloadProgress(modelId) {
    try {
      const res = await fetch(`http://localhost:8000/models/download-progress/${modelId}`);
      if (res.ok) {
        const progress = await res.json();
        setDownloadProgress(prev => ({ ...prev, [modelId]: progress }));
      }
    } catch (err) {
      console.error(`Failed to fetch progress for ${modelId}:`, err);
    }
  }

  async function downloadModel(modelId) {
    try {
      setLoading(true);
      const res = await fetch(`http://localhost:8000/models/download?model_id=${modelId}`, {
        method: "POST"
      });
      const data = await res.json();
      if (data.success) {
        // Start polling for progress
        const interval = setInterval(() => fetchDownloadProgress(modelId), 500);
        setTimeout(() => clearInterval(interval), 300000); // Stop after 5 minutes
      } else {
        alert(data.error || "Download failed");
      }
    } catch (err) {
      alert("Failed to start download");
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function selectModel(modelId) {
    try {
      setLoading(true);
      const res = await fetch(`http://localhost:8000/models/select?model_id=${modelId}`, {
        method: "POST"
      });
      const data = await res.json();
      if (data.success) {
        setActiveModelId(data.active_model_id);
        fetchModels();
      } else {
        alert(data.error || "Failed to select model");
      }
    } catch (err) {
      alert("Failed to select model");
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function deleteModel(modelId) {
    if (!confirm("Are you sure you want to delete this model file?")) return;

    try {
      const res = await fetch(`http://localhost:8000/models/${modelId}`, {
        method: "DELETE"
      });
      const data = await res.json();
      if (data.success) {
        fetchModels();
      } else {
        alert(data.error || "Failed to delete model");
      }
    } catch (err) {
      alert("Failed to delete model");
      console.error(err);
    }
  }

  async function addCustomModel() {
    if (!customModelUrl.trim()) {
      alert("Please enter a valid URL");
      return;
    }

    // Extract filename from URL
    const filename = customModelUrl.split("/").pop();
    const modelId = filename.replace(".gguf", "").toLowerCase();

    const modelInfo = {
      id: modelId,
      name: filename,
      url: customModelUrl,
      size: "Unknown",
      description: "Custom model"
    };

    try {
      const res = await fetch("http://localhost:8000/models/add-custom", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(modelInfo)
      });
      const data = await res.json();
      if (data.success) {
        setCustomModelUrl("");
        setShowAddCustom(false);
        fetchModels();
      } else {
        alert(data.error || "Failed to add custom model");
      }
    } catch (err) {
      alert("Failed to add custom model");
      console.error(err);
    }
  }

  const styles = {
    overlay: {
      position: "fixed",
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: "rgba(0, 0, 0, 0.7)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      zIndex: 1000
    },
    modal: {
      backgroundColor: theme.sidebarBg,
      color: theme.text,
      borderRadius: "12px",
      padding: "30px",
      maxWidth: "600px",
      width: "90%",
      maxHeight: "80vh",
      overflowY: "auto",
      boxShadow: "0 4px 20px rgba(0,0,0,0.3)"
    },
    header: {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      marginBottom: "20px"
    },
    title: {
      fontSize: "24px",
      fontWeight: "bold"
    },
    closeButton: {
      backgroundColor: "transparent",
      border: "none",
      color: theme.text,
      fontSize: "24px",
      cursor: "pointer"
    },
    modelCard: {
      backgroundColor: theme.chatItemBg,
      borderRadius: "8px",
      padding: "15px",
      marginBottom: "15px",
      border: `2px solid ${theme.buttonHover}`
    },
    modelCardActive: {
      backgroundColor: theme.chatItemBg,
      borderRadius: "8px",
      padding: "15px",
      marginBottom: "15px",
      border: `2px solid #4CAF50`
    },
    modelName: {
      fontSize: "16px",
      fontWeight: "bold",
      marginBottom: "5px"
    },
    modelDesc: {
      fontSize: "13px",
      color: theme.text,
      opacity: 0.7,
      marginBottom: "10px"
    },
    modelInfo: {
      display: "flex",
      gap: "15px",
      marginBottom: "10px",
      fontSize: "12px"
    },
    button: {
      padding: "8px 15px",
      borderRadius: "6px",
      border: "none",
      cursor: loading ? "not-allowed" : "pointer",
      marginRight: "8px",
      display: "inline-flex",
      alignItems: "center",
      gap: "5px",
      fontSize: "14px",
      opacity: loading ? 0.6 : 1
    },
    downloadButton: {
      backgroundColor: "#2196F3",
      color: "white"
    },
    selectButton: {
      backgroundColor: "#4CAF50",
      color: "white"
    },
    deleteButton: {
      backgroundColor: "#f44336",
      color: "white"
    },
    addCustomButton: {
      backgroundColor: theme.buttonBg,
      color: theme.text,
      padding: "10px",
      border: `1px solid ${theme.buttonHover}`,
      borderRadius: "6px",
      cursor: "pointer",
      marginTop: "10px",
      width: "100%",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      gap: "8px"
    },
    input: {
      width: "100%",
      padding: "10px",
      borderRadius: "6px",
      border: `1px solid ${theme.buttonHover}`,
      backgroundColor: theme.chatBg,
      color: theme.text,
      marginBottom: "10px",
      fontSize: "14px"
    },
    progressBar: {
      width: "100%",
      height: "8px",
      backgroundColor: theme.chatBg,
      borderRadius: "4px",
      overflow: "hidden",
      marginTop: "10px"
    },
    progressFill: (percent) => ({
      width: `${percent}%`,
      height: "100%",
      backgroundColor: "#4CAF50",
      transition: "width 0.3s"
    })
  };

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.modal} onClick={e => e.stopPropagation()}>
        <div style={styles.header}>
          <h2 style={styles.title}>Model Manager</h2>
          <button style={styles.closeButton} onClick={onClose}>×</button>
        </div>

        <div>
          {models.map(model => {
            const progress = downloadProgress[model.id];
            const isDownloading = progress && progress.status === "downloading";

            return (
              <div key={model.id} style={model.loaded ? styles.modelCardActive : styles.modelCard}>
                <div style={styles.modelName}>
                  {model.name}
                  {model.loaded && <FaCheck style={{ marginLeft: "10px", color: "#4CAF50" }} />}
                </div>
                <div style={styles.modelDesc}>{model.description}</div>
                <div style={styles.modelInfo}>
                  <span>Size: {model.size}</span>
                  {model.downloaded && model.file_size_mb && (
                    <span>Downloaded: {model.file_size_mb} MB</span>
                  )}
                  {model.downloaded && <span style={{ color: "#4CAF50" }}>✓ Downloaded</span>}
                </div>

                {isDownloading && (
                  <div>
                    <div style={{ fontSize: "12px", marginBottom: "5px" }}>
                      Downloading: {progress.progress?.toFixed(1)}%
                    </div>
                    <div style={styles.progressBar}>
                      <div style={styles.progressFill(progress.progress || 0)} />
                    </div>
                  </div>
                )}

                <div style={{ marginTop: "10px" }}>
                  {!model.downloaded && !isDownloading && (
                    <button
                      style={{ ...styles.button, ...styles.downloadButton }}
                      onClick={() => downloadModel(model.id)}
                      disabled={loading}
                    >
                      <FaDownload /> Download
                    </button>
                  )}

                  {model.downloaded && !model.loaded && (
                    <button
                      style={{ ...styles.button, ...styles.selectButton }}
                      onClick={() => selectModel(model.id)}
                      disabled={loading}
                    >
                      {loading ? <FaSpinner className="spin" /> : <FaCheck />} Select
                    </button>
                  )}

                  {model.loaded && (
                    <span style={{ color: "#4CAF50", fontSize: "14px", fontWeight: "bold" }}>
                      ● Active
                    </span>
                  )}

                  {model.downloaded && !model.loaded && (
                    <button
                      style={{ ...styles.button, ...styles.deleteButton }}
                      onClick={() => deleteModel(model.id)}
                      disabled={loading}
                    >
                      <FaTrash /> Delete
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {!showAddCustom && (
          <button style={styles.addCustomButton} onClick={() => setShowAddCustom(true)}>
            <FaPlus /> Add Custom Model URL
          </button>
        )}

        {showAddCustom && (
          <div style={{ marginTop: "20px", padding: "15px", backgroundColor: theme.chatItemBg, borderRadius: "8px" }}>
            <h3 style={{ marginBottom: "10px" }}>Add Custom GGUF Model</h3>
            <input
              style={styles.input}
              type="text"
              placeholder="https://huggingface.co/.../model.gguf"
              value={customModelUrl}
              onChange={e => setCustomModelUrl(e.target.value)}
            />
            <button
              style={{ ...styles.button, ...styles.selectButton, marginRight: "10px" }}
              onClick={addCustomModel}
            >
              Add Model
            </button>
            <button
              style={{ ...styles.button, ...styles.deleteButton }}
              onClick={() => { setShowAddCustom(false); setCustomModelUrl(""); }}
            >
              Cancel
            </button>
          </div>
        )}

        <style>{`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          .spin {
            animation: spin 1s linear infinite;
          }
        `}</style>
      </div>
    </div>
  );
}
