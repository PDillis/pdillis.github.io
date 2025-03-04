// Initialize TensorFlow.js visualization
function initTensorFlowVis(canvasId) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const controlsContainer = document.getElementById(`${canvasId}-controls`);
    
    // Create UI controls
    const createSlider = (id, label, min, max, step, defaultValue, onChange) => {
      const container = document.createElement('div');
      container.className = 'control-group';
      container.style.margin = '10px 0';
      
      const labelElement = document.createElement('label');
      labelElement.htmlFor = id;
      labelElement.textContent = label;
      labelElement.style.display = 'block';
      labelElement.style.marginBottom = '5px';
      
      const slider = document.createElement('input');
      slider.type = 'range';
      slider.id = id;
      slider.min = min;
      slider.max = max;
      slider.step = step;
      slider.value = defaultValue;
      slider.style.width = '100%';
      
      const valueDisplay = document.createElement('span');
      valueDisplay.textContent = defaultValue;
      valueDisplay.style.marginLeft = '10px';
      
      slider.addEventListener('input', (e) => {
        valueDisplay.textContent = e.target.value;
        onChange(parseFloat(e.target.value));
      });
      
      container.appendChild(labelElement);
      container.appendChild(slider);
      container.appendChild(valueDisplay);
      
      return container;
    };
    
    // Example: Simple LSTM generation viewer (latent space visualization)
    let noise = 0.5;
    let complexity = 0.5;
    let temperature = 1.0;
    
    // Add controls
    const noiseSlider = createSlider(
      `${canvasId}-noise`, 
      'Noise', 
      0, 1, 0.01, 
      noise, 
      (value) => {
        noise = value;
        drawLatentSpace();
      }
    );
    
    const complexitySlider = createSlider(
      `${canvasId}-complexity`, 
      'Complexity', 
      0, 1, 0.01, 
      complexity, 
      (value) => {
        complexity = value;
        drawLatentSpace();
      }
    );
    
    const temperatureSlider = createSlider(
      `${canvasId}-temperature`, 
      'Temperature', 
      0.1, 3, 0.1, 
      temperature, 
      (value) => {
        temperature = value;
        drawLatentSpace();
      }
    );
    
    controlsContainer.appendChild(noiseSlider);
    controlsContainer.appendChild(complexitySlider);
    controlsContainer.appendChild(temperatureSlider);
    
    // Create a simple model to generate data
    const createModel = async () => {
      const model = tf.sequential();
      
      model.add(tf.layers.dense({
        units: 16,
        inputShape: [2],
        activation: 'relu'
      }));
      
      model.add(tf.layers.dense({
        units: 32,
        activation: 'relu'
      }));
      
      model.add(tf.layers.dense({
        units: 2,
        activation: 'tanh'
      }));
      
      model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError'
      });
      
      return model;
    };
    
    let model;
    
    // Initialize the model
    (async () => {
      model = await createModel();
      drawLatentSpace();
    })();
    
    // Draw latent space visualization
    const drawLatentSpace = async () => {
      if (!model) return;
      
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const width = canvas.width;
      const height = canvas.height;
      
      // Generate points from latent space
      const numPoints = 500;
      const points = [];
      
      for (let i = 0; i < numPoints; i++) {
        // Generate random point in latent space with noise parameter
        const z1 = (Math.random() * 2 - 1) * noise;
        const z2 = (Math.random() * 2 - 1) * noise;
        
        // Get output from model
        const inputTensor = tf.tensor2d([[z1, z2]]);
        const outputTensor = model.predict(inputTensor);
        const output = await outputTensor.array();
        
        // Apply temperature and complexity
        const x = (output[0][0] * temperature * complexity * width / 2) + width / 2;
        const y = (output[0][1] * temperature * complexity * height / 2) + height / 2;
        
        points.push({ x, y, z1, z2 });
        
        // Cleanup tensors
        inputTensor.dispose();
        outputTensor.dispose();
      }
      
      // Draw points
      points.forEach(point => {
        // Map latent values to colors
        const r = Math.floor(((point.z1 + 1) / 2) * 255);
        const g = Math.floor(((point.z2 + 1) / 2) * 255);
        const b = Math.floor(((Math.abs(point.z1) + Math.abs(point.z2)) / 2) * 255);
        
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.beginPath();
        ctx.arc(point.x, point.y, 3 + 2 * Math.random(), 0, Math.PI * 2);
        ctx.fill();
      });
      
      // Draw info text
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(10, 10, 200, 60);
      ctx.fillStyle = 'white';
      ctx.font = '12px Arial';
      ctx.fillText(`Noise: ${noise.toFixed(2)}`, 20, 30);
      ctx.fillText(`Complexity: ${complexity.toFixed(2)}`, 20, 50);
      ctx.fillText(`Temperature: ${temperature.toFixed(2)}`, 20, 70);
    };
    
    // Resize handler
    const resizeCanvas = () => {
      canvas.width = canvas.clientWidth;
      drawLatentSpace();
    };
    
    // Initial setup
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
  }