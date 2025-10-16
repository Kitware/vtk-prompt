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
};
