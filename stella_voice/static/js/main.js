document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
    const statusSpan = document.getElementById('status');
    const scoreSpan = document.getElementById('score');
    const caseImage = document.getElementById('case-image');

    // Client-side state for the score
    let currentScore = 0;
    let currentAttempted = 0;

    let audioContext;
    let mediaStream;
    let scriptProcessor;
    let sourceNode;
    const TARGET_SAMPLE_RATE = 16000;
    const BUFFER_SIZE = 4096;

    // --- WebSocket Event Handlers ---
    socket.on('connect', () => {
        console.log('Connected to server. Starting session...');
        statusSpan.textContent = 'Connecting...';
        // Automatically start listening and trigger the session
        startListening().then(() => {
            socket.emit('start_session');
        });
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server.');
        statusSpan.textContent = 'Disconnected. Please refresh.';
        stopListening();
    });

    socket.on('update_status', (data) => {
        statusSpan.textContent = data.status;
        if (data.status.includes('Session Ended')) {
            stopListening();
        }
    });

    socket.on('update_case', (data) => {
        if (data.image_url) {
            caseImage.src = data.image_url;
            caseImage.style.display = 'block';
        }

        if (data.score !== undefined) {
            currentScore = data.score;
        }
        if (data.attempted !== undefined) {
            currentAttempted = data.attempted;
        }
        scoreSpan.textContent = `${currentScore} / ${currentAttempted}`;
    });

    socket.on('play_audio', (data) => {
        console.log("Received audio data to play.");

        if (scriptProcessor && sourceNode) {
            sourceNode.disconnect(scriptProcessor);
            console.log('Microphone processing PAUSED for AI speech.');
        }

        const audioData = data.audio;
        const audio = new Audio("data:audio/wav;base64," + audioData);
        audio.play();

        audio.onended = () => {
            console.log("AI speech finished.");
            if (scriptProcessor && sourceNode) {
                sourceNode.connect(scriptProcessor);
                console.log('Microphone processing RESUMED.');
            }
        };
    });

    // --- Audio Processing Logic ---
    const startListening = async () => {
        if (mediaStream) return; 

        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            sourceNode = audioContext.createMediaStreamSource(mediaStream);
            scriptProcessor = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);

            scriptProcessor.onaudioprocess = (event) => {
                const inputData = event.inputBuffer.getChannelData(0);
                const resampledData = downsampleBuffer(inputData, audioContext.sampleRate, TARGET_SAMPLE_RATE);
                socket.emit('audio_stream', Array.from(resampledData));
            };

            sourceNode.connect(scriptProcessor);
            statusSpan.textContent = ''; // Clear status on start

        } catch (err) {
            console.error('Error accessing microphone:', err);
            statusSpan.textContent = 'Microphone access denied.';
            alert('Could not access the microphone. Please check your browser permissions.');
        }
    };

    const stopListening = () => {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        if (scriptProcessor) {
            scriptProcessor.disconnect();
            scriptProcessor = null;
        }
        if (sourceNode) {
            sourceNode.disconnect();
            sourceNode = null;
        }
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
    };

    function downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
        if (outputSampleRate === inputSampleRate) {
            return buffer;
        }
        const sampleRateRatio = inputSampleRate / outputSampleRate;
        const newLength = Math.round(buffer.length / sampleRateRatio);
        const result = new Float32Array(newLength);
        let offsetResult = 0;
        let offsetBuffer = 0;
        while (offsetResult < result.length) {
            const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
            let accum = 0, count = 0;
            for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
                accum += buffer[i];
                count++;
            }
            result[offsetResult] = accum / count;
            offsetResult++;
            offsetBuffer = nextOffsetBuffer;
        }
        return result;
    }
});
