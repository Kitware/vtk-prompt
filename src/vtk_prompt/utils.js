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

  async exportConfig() {
    const text = await window.trame.trigger("save_config");
    if (text) this.download("vtk-prompt-config.yaml", text, "text/yaml");
  },
};
