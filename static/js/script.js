document.addEventListener("DOMContentLoaded", function () {
  const input = document.getElementById("fileInput");
  const label = document.getElementById("fileName");

  input.addEventListener("change", function () {
    if (input.files.length > 0) {
      label.innerText = `Selected file: ${input.files[0].name}`;
    } else {
      label.innerText = "No file selected";
    }
  });
});
