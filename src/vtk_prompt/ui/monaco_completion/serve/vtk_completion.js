// VTK/Python completion provider for the Monaco editor embedded by trame-code.
//
// trame-code does not expose its Monaco instance, so a tiny local patch to the
// trame-code bundle assigns window.__vtkmonaco and fires "vtk-monaco-ready".
// Here we register a Python completion provider against that Monaco instance.
// On each request it calls the in-process server trigger "jedi_complete" over
// the existing trame websocket and maps jedi results to Monaco suggestions.
(function () {
  if (window.__vtkCompletionInstalled) return;

  function kindMap(monaco) {
    var K = monaco.languages.CompletionItemKind;
    return {
      function: K.Function,
      method: K.Method,
      class: K.Class,
      instance: K.Variable,
      module: K.Module,
      keyword: K.Keyword,
      statement: K.Snippet,
      param: K.Variable,
      property: K.Property,
      path: K.File,
    };
  }

  function install(monaco) {
    if (!monaco || !monaco.languages || window.__vtkCompletionInstalled) return;
    window.__vtkCompletionInstalled = true;
    var KIND = kindMap(monaco);

    monaco.languages.registerCompletionItemProvider("python", {
      triggerCharacters: ["."],
      provideCompletionItems: async function (model, position) {
        var items = [];
        try {
          var code = model.getValue();
          var line = position.lineNumber; // 1-based, matches jedi
          var column = position.column - 1; // monaco col is 1-based; jedi wants 0-based
          items = await window.trame.trigger("jedi_complete", [code, line, column]);
        } catch (e) {
          items = [];
        }
        if (!items) items = [];

        var word = model.getWordUntilPosition(position);
        var range = {
          startLineNumber: position.lineNumber,
          endLineNumber: position.lineNumber,
          startColumn: word.startColumn,
          endColumn: word.endColumn,
        };

        return {
          suggestions: items.map(function (it) {
            return {
              label: it.label,
              kind: KIND[it.kind] || monaco.languages.CompletionItemKind.Text,
              insertText: it.label,
              detail: it.detail || "",
              range: range,
            };
          }),
        };
      },
    });
    console.debug("[vtk] Python/VTK completion provider registered");

    monaco.languages.registerHoverProvider("python", {
      provideHover: async function (model, position) {
        var info = null;
        try {
          var code = model.getValue();
          var line = position.lineNumber; // 1-based
          var column = position.column - 1; // 0-based for jedi
          info = await window.trame.trigger("jedi_hover", [code, line, column]);
        } catch (e) {
          info = null;
        }
        if (!info) return null;

        var contents = [];
        if (info.signatures && info.signatures.length) {
          contents.push({ value: "```python\n" + info.signatures.join("\n") + "\n```" });
        } else if (info.name) {
          contents.push({ value: "```python\n" + info.name + "\n```" });
        }
        if (info.prose) {
          // single newlines -> markdown hard breaks so the prose wraps readably
          contents.push({ value: info.prose.replace(/\n/g, "  \n") });
        }
        if (!contents.length) return null;

        var word = model.getWordAtPosition(position);
        var range = word
          ? {
              startLineNumber: position.lineNumber,
              endLineNumber: position.lineNumber,
              startColumn: word.startColumn,
              endColumn: word.endColumn,
            }
          : undefined;
        return { range: range, contents: contents };
      },
    });
    console.debug("[vtk] Python/VTK hover provider registered");
  }

  if (window.__vtkmonaco) install(window.__vtkmonaco);
  window.addEventListener("vtk-monaco-ready", function () {
    install(window.__vtkmonaco);
  });
})();
