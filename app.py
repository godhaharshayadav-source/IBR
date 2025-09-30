<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fashion Simulator</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&family=Orbitron:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #000000;
            --text: #E0E0E0;
            --accent: #00F6FF;
            --accent-dark: #008C99;
        }
        body {
            background-color: var(--bg);
            color: var(--text);
            font-family: 'Roboto Mono', monospace;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 80vh;
        }
        .font-orbitron { font-family: 'Orbitron', sans-serif; }
        .container {
            width: 100%;
            height: 100%;
            max-width: 1400px;
            max-height: 800px;
            background: radial-gradient(circle, rgba(1, 19, 34, 0.5) 0%, rgba(0,0,0,0) 70%), #000;
            border: 1px solid var(--accent-dark);
            box-shadow: 0 0 50px rgba(0, 246, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        .scene {
            position: absolute;
            inset: 0;
            opacity: 0;
            transition: opacity 1s ease-in-out;
            visibility: hidden;
        }
        .scene.active {
            opacity: 1;
            visibility: visible;
        }
        .vignette {
            position: absolute;
            inset: 0;
            box-shadow: inset 0 0 150px rgba(0,0,0,0.9);
            pointer-events: none;
        }
        .glow { text-shadow: 0 0 8px var(--accent); }
        .typewriter {
            overflow: hidden;
            white-space: nowrap;
            animation: typing 2.5s steps(40, end);
        }
        @keyframes typing { from { width: 0 } to { width: 100% } }
        
        /* Scene 1: Signal */
        #signal-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
        .signal-hotspot {
            position: absolute;
            width: 20px;
            height: 20px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 0 20px var(--accent);
            animation: pulse-signal 1.5s infinite;
        }
        @keyframes pulse-signal { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.3); } }

        /* Scene 2: Vision */
        .vision-card {
            backdrop-filter: blur(10px);
            background: rgba(0, 50, 60, 0.3);
            border: 1px solid var(--accent-dark);
            transition: all 0.5s ease;
        }
        
        /* Scene 3: Forecast */
        .map-region { fill: rgba(0, 140, 153, 0.2); stroke: var(--accent); stroke-width: 0.5; }
        .demand-wave {
            fill: var(--accent);
            opacity: 0.5;
            transform-origin: center;
            animation: expand-wave 4s infinite ease-out;
        }
        @keyframes expand-wave { 0% { r: 0; opacity: 0.6; } 100% { r: 100px; opacity: 0; } }

        /* Scene 4: Path */
        #path-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
        
        /* Scene 5: Impact */
        .impact-bar {
            background: linear-gradient(90deg, var(--accent-dark) 0%, var(--accent) 100%);
            animation: grow-bar 1.5s ease-out forwards;
            transform-origin: left;
        }
        @keyframes grow-bar { 0% { transform: scaleX(0); } 100% { transform: scaleX(1); } }

        .loader {
            width: 48px; height: 48px; border: 3px solid var(--accent);
            border-bottom-color: transparent; border-radius: 50%;
            display: inline-block; box-sizing: border-box;
            animation: rotation 1s linear infinite;
        }
        @keyframes rotation { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

    </style>
</head>
<body>
    <div class="container">
        <!-- SCENE 1: THE SIGNAL -->
        <div id="scene1" class="scene active flex flex-col items-center justify-center text-center">
            <canvas id="signal-canvas"></canvas>
            <div id="signal-content">
                <h1 class="font-orbitron text-4xl glow typewriter">FASHION SIMULATOR: ANALYZING SIGNALS...</h1>
                <p id="signal-instruction" class="mt-4 text-lg opacity-0 transition-opacity duration-1000">A new trend signal is emerging. Isolate it.</p>
            </div>
        </div>

        <!-- SCENE 2: THE VISION -->
        <div id="scene2" class="scene p-12 flex items-center justify-center">
             <div id="vision-loader" class="text-center">
                <div class="loader"></div>
                <p class="mt-4 font-orbitron glow">AI IS DREAMING...</p>
            </div>
            <div id="vision-content" class="hidden w-full h-full grid grid-cols-2 gap-8">
                 <div id="mood-board" class="vision-card p-6 flex flex-col justify-center items-center">
                    <img id="generated-image" src="https://placehold.co/500x500/000000/00F6FF?text=GENERATING..." class="w-full h-auto object-cover rounded-lg border-2 border-cyan-500/50 shadow-lg shadow-cyan-500/20"/>
                 </div>
                 <div class="flex flex-col justify-center text-left">
                     <h2 class="font-orbitron text-3xl glow" id="vision-title">--</h2>
                     <p class="mt-4 text-lg leading-relaxed" id="vision-description">--</p>
                     <button id="vision-next-btn" class="mt-8 w-1/2 p-3 font-orbitron bg-transparent border-2 border-cyan-400 text-cyan-400 hover:bg-cyan-400 hover:text-black transition-all duration-300">CALCULATE GLOBAL DEMAND</button>
                 </div>
            </div>
        </div>

        <!-- SCENE 3: THE FORECAST -->
        <div id="scene3" class="scene flex flex-col items-center justify-center p-8">
            <h2 class="font-orbitron text-3xl glow mb-4">DEMAND FORECAST: GLOBAL SPREAD</h2>
            <svg id="forecast-map" viewBox="0 0 1000 500" class="w-full h-auto max-h-[80%]">
                <path class="map-region" d="M123.5,142L2.5,214.5L1,248.5L25,263L13,313L29,322L55.5,310L99,313L101,327L128,318.5L145.5,326L164.5,301L181.5,321.5L200,317L213.5,296L232,305L268.5,274L288,279L298.5,264.5L317,272L329,248L321.5,223.5L339,219L338,203.5L347,200L352,192L348,154.5L323,139L299.5,116L275,100L252.5,50.5L238.5,38L221.5,41L212.5,31.5L189.5,19L170,1.5L127,1L99,19.5L62.5,28L39,59L8.5,108L37,137.5L88,142Z"/>
                <path class="map-region" d="M268.5,274L232,305L213.5,296L200,317L181.5,321.5L164.5,301L145.5,326L128,318.5L101,327L99,313L115,355.5L140,404L165.5,419L191,442.5L209,425L232.5,454.5L248.5,433.5L263.5,404.5L273,383.5L288,340L288,279Z"/>
                <path class="map-region" d="M366.5,274L364,302L345,327.5L356,350L331.5,384L337,419L362,448L388,443L405.5,414.5L421,424.5L424,394L440.5,373.5L426.5,349L445,328L428,295.5L418,274L403.5,263L379.5,262Z"/>
                <path class="map-region" d="M352,192L347,200L338,203.5L339,219L321.5,223.5L329,248L317,272L366.5,274L403.5,263L418,274L412,231L420.5,223.5L414.5,214L424.5,200L409,192.5L403.5,160.5L383,169L379,153L359.5,153L348,154.5Z"/>
                <path class="map-region" d="M418,274L428,295.5L445,328L426.5,349L440.5,373.5L462.5,361L480.5,337L500,320.5L511.5,302.5L506.5,279L481,259.5L453.5,261L440,248L424.5,252.5Z"/>
                <path class="map-region" d="M403.5,160.5L409,192.5L424.5,200L414.5,214L420.5,223.5L412,231L418,274L440,248L453.5,261L481,259.5L506.5,279L511.5,302.5L500,320.5L480.5,337L462.5,361L483,385.5L514,379.5L532.5,400.5L555.5,384L572.5,395L597,381.5L620,398.5L638,380.5L653.5,389L672.5,376.5L693.5,391.5L713,371.5L724,383.5L749.5,368.5L769.5,386.5L795,374.5L810.5,355.5L828.5,361L853,349L869.5,329L889.5,337.5L912,323L937.5,300L957.5,271L980,250L998.5,222L986,198.5L960.5,178L936.5,152.5L912,126L883.5,108.5L844.5,95.5L804.5,84L769,70.5L727,61L693,52L654.5,43L613.5,44.5L579.5,56L546,74.5L512.5,99.5L478,124.5L443.5,145.5L423,151.5Z"/>
                <circle cx="650" cy="100" r="10" class="demand-wave" style="animation-delay: 0s;"></circle>
                <circle cx="400" cy="180" r="10" class="demand-wave" style="animation-delay: 1s;"></circle>
                <circle cx="150" cy="180" r="10" class="demand-wave" style="animation-delay: 2s;"></circle>
            </svg>
            <div id="forecast-summary" class="absolute bottom-8 left-8 vision-card p-4 text-sm opacity-0 transition-opacity duration-1000">
                <p>PEAK DEMAND: <span class="font-bold glow">WEEK 3</span></p>
                <p>RECOMMENDED BATCH: <span class="font-bold glow">150,000 Units</span></p>
            </div>
             <button id="forecast-next-btn" class="absolute bottom-8 right-8 p-3 font-orbitron bg-transparent border-2 border-cyan-400 text-cyan-400 hover:bg-cyan-400 hover:text-black transition-all duration-300 opacity-0">OPTIMIZE SUPPLY CHAIN</button>
        </div>

        <!-- SCENE 4: THE PATH -->
        <div id="scene4" class="scene">
            <canvas id="path-canvas"></canvas>
            <div id="disruption-alert" class="hidden absolute top-8 right-8 vision-card p-4 max-w-sm">
                <h3 class="font-orbitron glow text-red-400">!! DISRUPTION DETECTED !!</h3>
                <p id="disruption-details" class="mt-2 text-sm">--</p>
            </div>
        </div>

        <!-- SCENE 5: THE IMPACT -->
        <div id="scene5" class="scene flex flex-col items-center justify-center p-12">
            <h2 class="font-orbitron text-4xl glow">SIMULATION OUTCOME: SUCCESS</h2>
            <div class="mt-12 w-full max-w-3xl space-y-6">
                <div>
                    <div class="flex justify-between items-baseline"><p>PROFIT GROWTH</p><p class="font-orbitron text-2xl glow">+32%</p></div>
                    <div class="w-full h-4 mt-1 bg-gray-800"><div class="h-full impact-bar"></div></div>
                </div>
                <div>
                    <div class="flex justify-between items-baseline"><p>TEXTILE WASTE AVOIDED</p><p class="font-orbitron text-2xl glow">18.2 Tonnes</p></div>
                    <div class="w-full h-4 mt-1 bg-gray-800"><div class="h-full impact-bar" style="width: 78%;"></div></div>
                </div>
                <div>
                    <div class="flex justify-between items-baseline"><p>CARBON EMISSIONS SAVED</p><p class="font-orbitron text-2xl glow">41.5 Tonnes CO₂e</p></div>
                    <div class="w-full h-4 mt-1 bg-gray-800"><div class="h-full impact-bar" style="width: 85%;"></div></div>
                </div>
            </div>
             <button id="impact-restart-btn" class="mt-12 p-3 font-orbitron bg-transparent border-2 border-cyan-400 text-cyan-400 hover:bg-cyan-400 hover:text-black transition-all duration-300">RUN NEW SIMULATION</button>
        </div>

        <div class="vignette"></div>
    </div>

<script>
document.addEventListener('DOMContentLoaded', () => {
    const scenes = document.querySelectorAll('.scene');
    let currentScene = 1;

    function switchScene(sceneNumber) {
        scenes.forEach(s => s.classList.remove('active'));
        const nextScene = document.getElementById(`scene${sceneNumber}`);
        if (nextScene) {
            nextScene.classList.add('active');
            currentScene = sceneNumber;
        }
    }

    // --- GEMINI API ---
    const API_KEY = "";
    const API_URL_TEXT = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${API_KEY}`;
    const API_URL_IMAGE = `https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key=${API_KEY}`;
    
    async function callGeminiText(prompt) {
        const payload = { contents: [{ parts: [{ text: prompt }] }] };
        const response = await fetch(API_URL_TEXT, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        if (!response.ok) throw new Error("API Error");
        const data = await response.json();
        return data.candidates[0].content.parts[0].text;
    }

    async function callGeminiImage(prompt) {
        const payload = { instances: [{ prompt }], parameters: { "sampleCount": 1 } };
        const response = await fetch(API_URL_IMAGE, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        if (!response.ok) throw new Error("Image API Error");
        const data = await response.json();
        return `data:image/png;base64,${data.predictions[0].bytesBase64Encoded}`;
    }

    // --- SCENE 1: SIGNAL ---
    const canvas = document.getElementById('signal-canvas');
    const ctx = canvas.getContext('2d');
    let particles = [];
    
    function setupSignalScene() {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        particles = [];
        for (let i = 0; i < 150; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3,
                radius: Math.random() * 1.5
            });
        }
        animateSignal();
        
        setTimeout(() => {
            const hotspot = document.createElement('div');
            hotspot.className = 'signal-hotspot';
            hotspot.style.left = '65%';
            hotspot.style.top = '40%';
            hotspot.onclick = () => {
                hotspot.remove();
                document.getElementById('signal-content').style.opacity = '0';
                runVisionScene();
            };
            document.getElementById('scene1').appendChild(hotspot);
            document.getElementById('signal-instruction').style.opacity = '1';
        }, 3000);
    }

    function animateSignal() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'rgba(0, 246, 255, 0.5)';
        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fill();
        });
        if (currentScene === 1) requestAnimationFrame(animateSignal);
    }
    
    // --- SCENE 2: VISION ---
    async function runVisionScene() {
        switchScene(2);
        const visionLoader = document.getElementById('vision-loader');
        const visionContent = document.getElementById('vision-content');
        
        try {
            const trendPrompt = "Name a futuristic, edgy fashion trend that is just emerging. Give it a single, cool-sounding, one-word name ending in 'core'. Then, in a new paragraph, describe this trend in 2-3 sentences.";
            const trendResponse = await callGeminiText(trendPrompt);
            const [trendName, trendDescription] = trendResponse.split('\n\n');

            const productName = "a futuristic jacket";
            const imagePrompt = `Editorial fashion photo of a ${productName} inspired by the "${trendName}" trend. Dark, edgy, holographic, neon details, on a high-fashion model, cinematic lighting.`;
            
            const [imageUrl] = await Promise.all([
                callGeminiImage(imagePrompt),
            ]);

            document.getElementById('vision-title').textContent = `${trendName.replace(/\*/g, '')}: The ${productName}`;
            document.getElementById('vision-description').textContent = trendDescription;
            document.getElementById('generated-image').src = imageUrl;

            visionLoader.style.display = 'none';
            visionContent.classList.remove('hidden');
        } catch (e) {
            visionLoader.innerHTML = `<p class="text-red-500">SIMULATION FAILED. Please refresh.</p>`;
            console.error(e);
        }
    }

    // --- SCENE 3: FORECAST ---
    function runForecastScene() {
        switchScene(3);
        setTimeout(() => {
            document.getElementById('forecast-summary').style.opacity = '1';
        }, 2000);
        setTimeout(() => {
            document.getElementById('forecast-next-btn').style.opacity = '1';
        }, 3000);
    }

    // --- SCENE 4: PATH ---
    const pathCanvas = document.getElementById('path-canvas');
    const pathCtx = pathCanvas.getContext('2d');
    let supplyNodes = [];
    let connections = [];
    let cargo = {x: 0, y: 0, targetIndex: 1, speed: 1};
    let disruption = false;

    function setupPathScene() {
        pathCanvas.width = pathCanvas.offsetWidth;
        pathCanvas.height = pathCanvas.offsetHeight;
        supplyNodes = [
            { x: 100, y: pathCanvas.height / 2, label: "Materials" },
            { x: 350, y: pathCanvas.height / 2 - 100, label: "Factory A" },
            { x: 350, y: pathCanvas.height / 2 + 100, label: "Factory B" },
            { x: 700, y: pathCanvas.height / 2, label: "Distribution" },
            { x: pathCanvas.width - 100, y: pathCanvas.height / 2, label: "Markets" }
        ];
        connections = [ [0,1], [0,2], [1,3], [2,3], [3,4] ];
        cargo.x = supplyNodes[0].x;
        cargo.y = supplyNodes[0].y;
        disruption = false;
        document.getElementById('disruption-alert').classList.add('hidden');
        
        setTimeout(async () => {
            disruption = true;
            try {
                const alertPrompt = "Generate a concise, urgent supply chain alert. A major shipping lane is unexpectedly closed due to a sudden storm. Mention the problem and state that the AI is autonomously rerouting logistics.";
                const alertText = await callGeminiText(alertPrompt);
                document.getElementById('disruption-details').textContent = alertText;
                document.getElementById('disruption-alert').classList.remove('hidden');
            } catch (e) { console.error(e); }
        }, 5000);

        animatePath();
    }

    function animatePath() {
        pathCtx.clearRect(0, 0, pathCanvas.width, pathCanvas.height);
        
        // Draw connections
        pathCtx.strokeStyle = 'rgba(0, 246, 255, 0.3)';
        pathCtx.lineWidth = 1;
        connections.forEach(c => {
            pathCtx.beginPath();
            pathCtx.moveTo(supplyNodes[c[0]].x, supplyNodes[c[0]].y);
            pathCtx.lineTo(supplyNodes[c[1]].x, supplyNodes[c[1]].y);
            pathCtx.stroke();
        });
        
        // Disruption effect
        if(disruption) {
             pathCtx.strokeStyle = '#F87171';
             pathCtx.lineWidth = 3;
             pathCtx.beginPath();
             pathCtx.moveTo(supplyNodes[1].x, supplyNodes[1].y);
             pathCtx.lineTo(supplyNodes[3].x, supplyNodes[3].y);
             pathCtx.stroke();
        }

        // Draw nodes
        supplyNodes.forEach(node => {
            pathCtx.fillStyle = 'rgba(0, 246, 255, 0.8)';
            pathCtx.beginPath();
            pathCtx.arc(node.x, node.y, 10, 0, Math.PI * 2);
            pathCtx.fill();
            pathCtx.fillStyle = '#fff';
            pathCtx.textAlign = 'center';
            pathCtx.fillText(node.label, node.x, node.y + 25);
        });

        // Move cargo
        let targetNode = supplyNodes[cargo.targetIndex];
        let dx = targetNode.x - cargo.x;
        let dy = targetNode.y - cargo.y;
        let dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < cargo.speed) {
            if(cargo.targetIndex === 1 && disruption) {
                cargo.targetIndex = 2; // Reroute
            } else if (cargo.targetIndex < 4) {
                 cargo.targetIndex = cargo.targetIndex === 1 || cargo.targetIndex === 2 ? 3 : cargo.targetIndex + 1;
            } else {
                 switchScene(5);
                 return;
            }
        } else {
            cargo.x += (dx / dist) * cargo.speed;
            cargo.y += (dy / dist) * cargo.speed;
        }

        pathCtx.fillStyle = '#FFD700';
        pathCtx.beginPath();
        pathCtx.arc(cargo.x, cargo.y, 6, 0, Math.PI * 2);
        pathCtx.fill();
        pathCtx.shadowColor = '#FFD700';
        pathCtx.shadowBlur = 15;
        pathCtx.fill();
        pathCtx.shadowBlur = 0;

        if(currentScene === 4) requestAnimationFrame(animatePath);
    }


    // --- EVENT LISTENERS ---
    document.getElementById('vision-next-btn').addEventListener('click', runForecastScene);
    document.getElementById('forecast-next-btn').addEventListener('click', () => {
        switchScene(4);
        setupPathScene();
    });
    document.getElementById('impact-restart-btn').addEventListener('click', () => {
        // Reset and restart the simulation
        const hotspot = document.querySelector('.signal-hotspot');
        if (hotspot) hotspot.remove();
        document.getElementById('signal-content').style.opacity = '1';
        document.getElementById('vision-content').classList.add('hidden');
        document.getElementById('vision-loader').style.display = 'block';
        document.getElementById('generated-image').src = 'https://placehold.co/500x500/000000/00F6FF?text=GENERATING...';
        document.getElementById('vision-title').textContent = '--';
        document.getElementById('vision-description').textContent = '--';
        document.getElementById('forecast-summary').style.opacity = '0';
        document.getElementById('forecast-next-btn').style.opacity = '0';
        
        switchScene(1);
        setupSignalScene();
    });

    // --- INITIALIZATION ---
    setupSignalScene();
});
</script>
</body>
</html>

