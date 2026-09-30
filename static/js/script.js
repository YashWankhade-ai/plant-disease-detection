// PlantCare AI - Frontend Interactive Scripts

document.addEventListener('DOMContentLoaded', function () {
  // Drop Zone and File Upload Elements
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('imageUploadInput');
  const previewContainer = document.getElementById('previewContainer');
  const previewImage = document.getElementById('previewImage');
  const fileNameDisplay = document.getElementById('fileNameDisplay');
  const removeFileBtn = document.getElementById('removeFileBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const uploadForm = document.getElementById('uploadForm');
  const loadingOverlay = document.getElementById('loadingOverlay');

  // Webcam elements
  const openCameraBtn = document.getElementById('openCameraBtn');
  const cameraModal = document.getElementById('cameraModal');
  const cameraVideo = document.getElementById('cameraVideo');
  const capturePhotoBtn = document.getElementById('capturePhotoBtn');
  const closeCameraBtn = document.getElementById('closeCameraBtn');
  let cameraStream = null;

  if (dropZone && fileInput) {
    // Click on drop zone triggers file input
    dropZone.addEventListener('click', function (e) {
      if (e.target !== removeFileBtn && !e.target.closest('#removeFileBtn')) {
        fileInput.click();
      }
    });

    // Drag over effect
    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
      });
    });

    // Handle dropped files
    dropZone.addEventListener('drop', function (e) {
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        fileInput.files = files;
        handleFileSelection(files[0]);
      }
    });

    // Handle input change
    fileInput.addEventListener('change', function () {
      if (fileInput.files.length > 0) {
        handleFileSelection(fileInput.files[0]);
      }
    });
  }

  function handleFileSelection(file) {
    if (!file) return;

    // Validate type
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      alert('Please upload a valid image file (.jpg, .jpeg, .png, .webp)');
      return;
    }

    // Validate size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
      alert('Image file size exceeds 16MB limit.');
      return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
      if (previewImage) {
        previewImage.src = e.target.result;
      }
      if (previewContainer) {
        previewContainer.style.display = 'block';
      }
      if (fileNameDisplay) {
        fileNameDisplay.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      }
      if (analyzeBtn) {
        analyzeBtn.disabled = false;
      }
    };
    reader.readAsDataURL(file);
  }

  // Remove previewed file
  if (removeFileBtn) {
    removeFileBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      if (fileInput) fileInput.value = '';
      if (previewImage) previewImage.src = '';
      if (previewContainer) previewContainer.style.display = 'none';
      if (analyzeBtn) analyzeBtn.disabled = true;
    });
  }

  // Form submit loading spinner
  if (uploadForm) {
    uploadForm.addEventListener('submit', function (e) {
      if (!fileInput.files || fileInput.files.length === 0) {
        e.preventDefault();
        alert('Please select or capture a plant leaf image first.');
        return;
      }
      if (loadingOverlay) {
        loadingOverlay.classList.remove('d-none');
      }
    });
  }

  // Camera integration
  if (openCameraBtn && cameraVideo) {
    openCameraBtn.addEventListener('click', async function () {
      try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
        });
        cameraVideo.srcObject = cameraStream;
        const modal = new bootstrap.Modal(cameraModal);
        modal.show();
      } catch (err) {
        console.error('Webcam access error:', err);
        alert('Unable to access camera: ' + err.message + '\nPlease check permissions or upload an image file.');
      }
    });

    if (capturePhotoBtn) {
      capturePhotoBtn.addEventListener('click', function () {
        const canvas = document.createElement('canvas');
        canvas.width = cameraVideo.videoWidth || 640;
        canvas.height = cameraVideo.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(function (blob) {
          const file = new File([blob], 'camera_leaf_scan.jpg', { type: 'image/jpeg' });
          const dataTransfer = new DataTransfer();
          dataTransfer.items.add(file);
          fileInput.files = dataTransfer.files;
          handleFileSelection(file);

          // Stop camera stream & close modal
          stopCamera();
          const modalInstance = bootstrap.Modal.getInstance(cameraModal);
          if (modalInstance) modalInstance.hide();
        }, 'image/jpeg', 0.95);
      });
    }

    if (closeCameraBtn) {
      closeCameraBtn.addEventListener('click', function () {
        stopCamera();
      });
    }

    if (cameraModal) {
      cameraModal.addEventListener('hidden.bs.modal', function () {
        stopCamera();
      });
    }
  }

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      cameraStream = null;
    }
  }

  // Supplements category filter
  const filterBtns = document.querySelectorAll('.supplement-filter-btn');
  const supplementItems = document.querySelectorAll('.supplement-item');

  if (filterBtns.length > 0 && supplementItems.length > 0) {
    filterBtns.forEach(btn => {
      btn.addEventListener('click', function () {
        filterBtns.forEach(b => b.classList.remove('active', 'btn-success'));
        filterBtns.forEach(b => b.classList.add('btn-outline-secondary'));

        this.classList.add('active', 'btn-success');
        this.classList.remove('btn-outline-secondary');

        const category = this.getAttribute('data-filter');

        supplementItems.forEach(item => {
          if (category === 'all' || item.getAttribute('data-category') === category) {
            item.style.display = 'block';
          } else {
            item.style.display = 'none';
          }
        });
      });
    });
  }
});
