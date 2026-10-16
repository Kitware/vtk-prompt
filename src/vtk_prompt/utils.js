window.trame.utils.vtk_prompt = {
  rules: {
    json_file(obj) {
      if (
        obj &&
        (obj.type !== "application/json" || !obj.name.endsWith(".json"))
      ) {
        return "Invalid file type";
      }
      return true;
    },
  },

  download(name, text, mime) {
    const blob = new Blob([text], { type: mime || "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  },

  sanitize(name) {
    return (name || "conversation").replace(/[^\w.-]+/g, "_").slice(0, 80);
  },

  async exportSession(id, title) {
    const text = await window.trame.trigger("export_session", [id]);
    if (text) {
      this.download(this.sanitize(title) + ".json", text, "application/json");
    }
  },

  // One file per conversation, so the result re-imports through the existing
  // multi-file import path. Sequential with a small gap: firing N downloads at
  // once gets some of them dropped, and Chrome will ask permission for the set.
  async exportSessions(ids, sessions) {
    const titles = {};
    (sessions || []).forEach((s) => {
      titles[s.id] = s.title;
    });
    for (const id of ids || []) {
      await this.exportSession(id, titles[id] || id);
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  },

  async exportConfig() {
    const text = await window.trame.trigger("save_config");
    if (text) this.download("vtk-prompt-config.yaml", text, "text/yaml");
  },

  // The react client has no file-input widget that serializes File objects
  // into state the way vuetify's VFileInput does, so the reading happens here
  // and the result is handed to the same python handler.
  async uploadFiles(fileList, triggerName) {
    const files = await Promise.all(
      Array.from(fileList || []).map(
        (file) =>
          new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = () =>
              resolve({
                name: file.name,
                size: file.size,
                type: file.type || "application/json",
                content: reader.result,
              });
            reader.onerror = () => resolve(null);
            reader.readAsText(file);
          }),
      ),
    );
    const valid = files.filter(Boolean);
    if (valid.length) {
      await window.trame.trigger(triggerName, [valid]);
    }
  },
};
