import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";

function convertMermaidCodeBlocks() {
  const codeBlocks = document.querySelectorAll("pre > code.language-mermaid, pre > code.mermaid");

  for (const code of codeBlocks) {
    const pre = code.closest("pre");
    if (!pre) continue;

    const container = document.createElement("div");
    container.className = "mermaid";
    container.textContent = code.textContent || "";

    pre.replaceWith(container);
  }
}

async function renderMermaid() {
  convertMermaidCodeBlocks();

  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
  });

  await mermaid.run({
    querySelector: ".mermaid",
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    renderMermaid();
  });
} else {
  renderMermaid();
}
