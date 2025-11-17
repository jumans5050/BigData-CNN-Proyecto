const MODEL_URL  = "model/best.onnx";
const LABELS_URL = "model/labels.txt";
const SCORE_TH   = 0.45;

let session = null;
let classNames = [];

const fileInput  = document.getElementById("fileInput");
const btnRun     = document.getElementById("btnRun");
const statusEl   = document.getElementById("status");
const imgEl      = document.getElementById("img");
const canvas     = document.getElementById("canvas");
const ctx        = canvas.getContext("2d");
const countsBody = document.getElementById("countsBody");

async function init() {
  try {
    statusEl.textContent = "Cargando modelo ONNX...";
    
    const response = await fetch(MODEL_URL);
    const arrayBuffer = await response.arrayBuffer();
    
    session = await ort.InferenceSession.create(arrayBuffer, {
      executionProviders: ["webgl", "cpu"]
    });

    const r = await fetch(LABELS_URL);
    if (!r.ok) throw new Error("Error al cargar labels: " + r.status);
    classNames = (await r.text()).trim().split("\n").map(s => s.trim());

    statusEl.textContent = "Modelo cargado. Seleccione una imagen.";
    btnRun.disabled = false;
  } catch (err) {
    console.error("Error al cargar modelo:", err);
    statusEl.textContent = "Error cargando modelo. Revise la consola.";
  }
}

fileInput.addEventListener("change", e => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = ev => {
    imgEl.onload = () => {
      canvas.width  = imgEl.naturalWidth;
      canvas.height = imgEl.naturalHeight;
      ctx.drawImage(imgEl, 0, 0);
      statusEl.textContent = "Imagen lista. Presione 'Detectar objetos'.";
    };
    imgEl.src = ev.target.result;
  };
  reader.readAsDataURL(file);
});

btnRun.addEventListener("click", async () => {
  if (!session) {
    statusEl.textContent = "Modelo no cargado.";
    return;
  }
  if (!imgEl.src) {
    statusEl.textContent = "Debe cargar una imagen primero.";
    return;
  }

  btnRun.disabled = true;
  statusEl.textContent = "Procesando imagen...";
  
  try {
    await detectar();
    statusEl.textContent = "Deteccion completada.";
  } catch (err) {
    console.error("Error en deteccion:", err);
    statusEl.textContent = "Error durante la deteccion.";
  } finally {
    btnRun.disabled = false;
  }
});

async function detectar() {
  canvas.width  = imgEl.naturalWidth;
  canvas.height = imgEl.naturalHeight;
  ctx.drawImage(imgEl, 0, 0);

  // Preparar imagen: redimensionar a 640x640 y normalizar
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 640;
  tempCanvas.height = 640;
  const tempCtx = tempCanvas.getContext("2d");
  
  tempCtx.drawImage(imgEl, 0, 0, 640, 640);
  const imageData = tempCtx.getImageData(0, 0, 640, 640);
  const data = imageData.data;

  // Convertir a tensor (NCHW format: 1x3x640x640)
  const tensorData = new Float32Array(1 * 3 * 640 * 640);
  let idx = 0;
  
  for (let i = 0; i < data.length; i += 4) {
    tensorData[idx++] = data[i] / 255.0;     // R
  }
  for (let i = 0; i < data.length; i += 4) {
    tensorData[idx++] = data[i + 1] / 255.0; // G
  }
  for (let i = 0; i < data.length; i += 4) {
    tensorData[idx++] = data[i + 2] / 255.0; // B
  }

  const tensor = new ort.Tensor("float32", tensorData, [1, 3, 640, 640]);
  const feeds = { images: tensor };

  let output;
  try {
    output = await session.run(feeds);
  } catch (err) {
    console.error("Error en inferencia:", err);
    statusEl.textContent = "Error en inferencia.";
    return;
  }

  // Obtener salida del modelo
  let detections = [];
  try {
    const outputData = output.output0.data;
    const outputShape = output.output0.dims;
    
    // Formato YOLO: [1, 10, 8400] -> [x, y, w, h, conf, class0-5]
    const numDetections = outputShape[2];
    const numOutputs = outputShape[1];
    
    for (let i = 0; i < numDetections; i++) {
      const conf = outputData[4 * numDetections + i];
      
      if (conf >= SCORE_TH) {
        const x = outputData[0 * numDetections + i];
        const y = outputData[1 * numDetections + i];
        const w = outputData[2 * numDetections + i];
        const h = outputData[3 * numDetections + i];
        
        let maxScore = 0;
        let classId = 0;
        for (let c = 0; c < 6; c++) {
          const score = outputData[(5 + c) * numDetections + i];
          if (score > maxScore) {
            maxScore = score;
            classId = c;
          }
        }
        
        // Convertir de centro a esquinas
        const xmin = (x - w / 2) / 640;
        const ymin = (y - h / 2) / 640;
        const xmax = (x + w / 2) / 640;
        const ymax = (y + h / 2) / 640;
        
        detections.push({
          box: [ymin, xmin, ymax, xmax],
          score: conf,
          classId: classId
        });
      }
    }
  } catch (err) {
    console.error("Error leyendo salidas:", err);
    statusEl.textContent = "Error leyendo salidas del modelo.";
    return;
  }

  dibujar(detections);
  actualizarTabla(detections);
}

function dibujar(dets) {
  ctx.drawImage(imgEl, 0, 0);
  ctx.lineWidth = 2;
  ctx.font = "14px system-ui";

  dets.forEach(d => {
    const [ymin, xmin, ymax, xmax] = d.box;
    const x = xmin * canvas.width;
    const y = ymin * canvas.height;
    const w = (xmax - xmin) * canvas.width;
    const h = (ymax - ymin) * canvas.height;

    ctx.strokeStyle = "#00ffff";
    ctx.strokeRect(x, y, w, h);

    const name = classNames[d.classId] || "id:" + d.classId;
    const label = name + " " + (d.score * 100).toFixed(1) + "%";
    const tw = ctx.measureText(label).width + 6;

    ctx.fillStyle = "#00ffff";
    ctx.fillRect(x, y - 18, tw, 18);
    ctx.fillStyle = "#000";
    ctx.fillText(label, x + 3, y - 5);
  });
}

function actualizarTabla(dets) {
  const counts = {};
  dets.forEach(d => {
    const name = classNames[d.classId] || "id:" + d.classId;
    counts[name] = (counts[name] || 0) + 1;
  });

  countsBody.innerHTML = "";
  Object.entries(counts).forEach(([name, c]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = "<td>" + name + "</td><td>" + c + "</td>";
    countsBody.appendChild(tr);
  });
}

window.addEventListener('load', init);