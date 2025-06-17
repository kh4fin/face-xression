const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const result = document.getElementById("result");

navigator.mediaDevices
  .getUserMedia({ video: true })
  .then((stream) => (video.srcObject = stream));

function capture() {
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataURL = canvas.toDataURL("image/jpeg");

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataURL }),
  })
    .then((response) => response.json())
    .then((data) => {
      result.textContent = `Ekspresi: ${data.label} (Confidence: ${(
        data.confidence * 100
      ).toFixed(2)}%)`;
    })
    .catch((err) => {
      result.textContent = "Gagal memproses gambar";
      console.error(err);
    });
}
