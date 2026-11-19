const arena = document.getElementById("arena");
const fileInput = document.getElementById("file");
const dropzone = document.getElementById("dropzone");
const statusEl = document.getElementById("status");
const again = document.getElementById("again");
const vs = document.getElementById("vs");
const preview = document.getElementById("preview");
const previewImg = document.getElementById("preview-img");
const sides = { Duck: document.getElementById("side-duck"), Swan: document.getElementById("side-swan") };
const badges = { Duck: document.getElementById("badge-duck"), Swan: document.getElementById("badge-swan") };

const MIN_THINK_MS = 1400; // let the "thinking" animation play a little

function reset() {
  arena.dataset.state = "idle";
  Object.values(sides).forEach((s) => s.classList.remove("win", "lose"));
  Object.values(badges).forEach((b) => (b.textContent = ""));
  preview.hidden = true;
  vs.hidden = false;
  again.hidden = true;
  statusEl.textContent = "";
  fileInput.value = "";
}

async function classify(file) {
  if (!file || !file.type.startsWith("image/")) {
    statusEl.textContent = "Please choose an image file.";
    return;
  }
  reset();
  previewImg.src = URL.createObjectURL(file);
  preview.hidden = false;
  arena.dataset.state = "thinking";
  statusEl.textContent = "The CNN is thinking...";

  const body = new FormData();
  body.append("file", file);
  try {
    const [res] = await Promise.all([
      fetch("/predict", { method: "POST", body }),
      new Promise((r) => setTimeout(r, MIN_THINK_MS)),
    ]);
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const { label, confidence } = await res.json();
    const loser = label === "Duck" ? "Swan" : "Duck";
    arena.dataset.state = "result";
    sides[label].classList.add("win");
    sides[loser].classList.add("lose");
    const pctText = confidence >= 0.9995 ? "> 99.9%" : `${(confidence * 100).toFixed(1)}%`;
    badges[label].textContent = `✓ ${pctText}`;
    statusEl.textContent = `It's a ${label.toLowerCase()}! (confidence ${pctText})`;
  } catch (err) {
    arena.dataset.state = "idle";
    statusEl.textContent = `Error: ${err.message}`;
  }
  again.hidden = false;
}

fileInput.addEventListener("change", () => classify(fileInput.files[0]));
again.addEventListener("click", reset);

["dragenter", "dragover"].forEach((ev) =>
  document.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  document.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
document.addEventListener("drop", (e) => classify(e.dataTransfer.files[0]));
