window.trame = window.trame || {};
window.trame.utils = window.trame.utils || {};

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
  copy_to_clipboard: function (text) {
    if (navigator && navigator.clipboard) {
      navigator.clipboard.writeText(text).catch((err) => {
        console.error("Failed to copy to clipboard:", err);
      });
    } else {
      console.error("Clipboard API not available");
    }
  },
  download_file: function (content, filename, mimeType) {
    mimeType = mimeType || "text/plain";
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  },
  download_generated_code: function (trame) {
    trame.trigger("download_generated_code").then((code) => {
      if (code) {
        window.trame.utils.vtk_prompt.download_file(
          code,
          "vtk_generated.py",
          "text/x-python",
        );
      }
    });
  },
};
