const { createOpenAiGenerativeClient, getOpenAiGenerativeModel } = require('./openAiClient.js');
const { createGeminiClient, getGeminiGenerativeModel, createGeminiChat } = require('./googleGeminiClient.js');

/**
 * Creates an Ollama client for local AI models
 * @param {string} baseUrl - The Ollama server URL (default: http://localhost:11434)
 * @returns {object} The Ollama client
 */
function createOllamaClient(baseUrl = 'http://localhost:11434') {
    return {
        baseUrl,
        async generateContent(model, prompt, options = {}) {
            const response = await fetch(`${baseUrl}/api/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: model || 'llama2',
                    prompt,
                    stream: false,
                    ...options
                }),
            });
            
            if (!response.ok) {
                throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        },
        
        async streamGenerate(model, prompt, options = {}) {
            const response = await fetch(`${baseUrl}/api/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: model || 'llama2',
                    prompt,
                    stream: true,
                    ...options
                }),
            });
            
            if (!response.ok) {
                throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
            }
            
            return response;
        }
    };
}

/**
 * Creates an AI client based on the provider
 * @param {string} apiKey - The API key (not needed for Ollama)
 * @param {string} provider - The provider ('openai', 'gemini', or 'ollama')
 * @param {string} ollamaUrl - The Ollama server URL (optional)
 * @returns {object} The AI client
 */
function createAIClient(apiKey, provider = 'openai', ollamaUrl = 'http://localhost:11434') {
    switch (provider) {
        case 'openai':
            return createOpenAiGenerativeClient(apiKey);
        case 'gemini':
            return createGeminiClient(apiKey);
        case 'ollama':
            return createOllamaClient(ollamaUrl);
        default:
            throw new Error(`Unsupported AI provider: ${provider}`);
    }
}

/**
 * Gets a generative model based on the provider
 * @param {object} client - The AI client
 * @param {string} provider - The provider ('openai', 'gemini', or 'ollama')
 * @param {string} model - The model name (optional)
 * @returns {object} The model object
 */
function getGenerativeModel(client, provider = 'openai', model) {
    switch (provider) {
        case 'openai':
            return getOpenAiGenerativeModel(client, model || 'gpt-4.1');
        case 'gemini':
            return getGeminiGenerativeModel(client, model || 'gemini-2.5-flash');
        case 'ollama':
            return {
                generateContent: async (prompt) => {
                    const result = await client.generateContent(model || 'llama2', prompt);
                    return { response: { text: () => result.response } };
                }
            };
        default:
            throw new Error(`Unsupported AI provider: ${provider}`);
    }
}

/**
 * Makes a chat completion request based on the provider
 * @param {object} params - Request parameters
 * @returns {Promise<object>} The completion response
 */
async function makeChatCompletion({ apiKey, provider = 'openai', messages, temperature = 0.7, maxTokens = 1024, model, stream = false, ollamaUrl = 'http://localhost:11434' }) {
    if (provider === 'openai') {
        const fetchUrl = 'https://api.openai.com/v1/chat/completions';
        const response = await fetch(fetchUrl, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: model || 'gpt-4.1',
                messages,
                temperature,
                max_tokens: maxTokens,
                stream,
            }),
        });

        if (!response.ok) {
            throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
        }

        if (stream) {
            return response;
        }

        const result = await response.json();
        return {
            content: result.choices[0].message.content.trim(),
            raw: result
        };
    } else if (provider === 'gemini') {
        const client = createGeminiClient(apiKey);
        const genModel = getGeminiGenerativeModel(client, model || 'gemini-2.5-flash');
        
        // Convert OpenAI format messages to Gemini format
        const parts = [];
        for (const message of messages) {
            if (message.role === 'system') {
                parts.push(message.content);
            } else if (message.role === 'user') {
                if (typeof message.content === 'string') {
                    parts.push(message.content);
                } else if (Array.isArray(message.content)) {
                    // Handle multimodal content
                    for (const part of message.content) {
                        if (part.type === 'text') {
                            parts.push(part.text);
                        } else if (part.type === 'image_url' && part.image_url?.url) {
                            // Extract base64 data from data URL
                            const base64Match = part.image_url.url.match(/^data:(.+);base64,(.+)$/);
                            if (base64Match) {
                                parts.push({
                                    inlineData: {
                                        mimeType: base64Match[1],
                                        data: base64Match[2]
                                    }
                                });
                            }
                        }
                    }
                }
            }
        }
        
        const result = await genModel.generateContent(parts);
        return {
            content: result.response.text(),
            raw: result
        };
    } else if (provider === 'ollama') {
        // Convert OpenAI chat format to Ollama format
        let prompt = '';
        for (const message of messages) {
            if (message.role === 'system') {
                prompt += `System: ${message.content}\n\n`;
            } else if (message.role === 'user') {
                const content = typeof message.content === 'string' ? message.content : 
                    message.content.find(part => part.type === 'text')?.text || '';
                prompt += `User: ${content}\n\n`;
            } else if (message.role === 'assistant') {
                prompt += `Assistant: ${message.content}\n\n`;
            }
        }
        prompt += 'Assistant:';

        const response = await fetch(`${ollamaUrl}/api/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: model || 'llama2',
                prompt,
                stream: false,
                options: {
                    temperature,
                    num_predict: maxTokens
                }
            }),
        });

        if (!response.ok) {
            throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
        }

        const result = await response.json();
        return {
            content: result.response?.trim() || '',
            raw: result
        };
    } else {
        throw new Error(`Unsupported AI provider: ${provider}`);
    }
}

/**
 * Makes a chat completion request with Portkey support
 * @param {object} params - Request parameters including Portkey options
 * @returns {Promise<object>} The completion response
 */
async function makeChatCompletionWithPortkey({ 
    apiKey, 
    provider = 'openai', 
    messages, 
    temperature = 0.7, 
    maxTokens = 1024, 
    model, 
    usePortkey = false,
    portkeyVirtualKey = null 
}) {
    if (!usePortkey) {
        return makeChatCompletion({ apiKey, provider, messages, temperature, maxTokens, model });
    }
    
    // Portkey is only supported for OpenAI currently
    if (provider !== 'openai') {
        console.warn('Portkey is only supported for OpenAI provider, falling back to direct API');
        return makeChatCompletion({ apiKey, provider, messages, temperature, maxTokens, model });
    }
    
    const fetchUrl = 'https://api.portkey.ai/v1/chat/completions';
    const response = await fetch(fetchUrl, {
        method: 'POST',
        headers: {
            'x-portkey-api-key': 'gRv2UGRMq6GGLJ8aVEB4e7adIewu',
            'x-portkey-virtual-key': portkeyVirtualKey || apiKey,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: model || 'gpt-4.1',
            messages,
            temperature,
            max_tokens: maxTokens,
        }),
    });

    if (!response.ok) {
        throw new Error(`Portkey API error: ${response.status} ${response.statusText}`);
    }

    const result = await response.json();
    return {
        content: result.choices[0].message.content.trim(),
        raw: result
    };
}

/**
 * Makes a streaming chat completion request
 * @param {object} params - Request parameters
 * @returns {Promise<Response>} The streaming response
 */
async function makeStreamingChatCompletion({ apiKey, provider = 'openai', messages, temperature = 0.7, maxTokens = 1024, model }) {
    if (provider === 'openai') {
        const fetchUrl = 'https://api.openai.com/v1/chat/completions';
        const response = await fetch(fetchUrl, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: model || 'gpt-4.1',
                messages,
                temperature,
                max_tokens: maxTokens,
                stream: true,
            }),
        });

        if (!response.ok) {
            throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
        }

        return response;
    } else if (provider === 'gemini') {
        console.log('[AIProviderService] Starting Gemini streaming request');
        // Gemini streaming requires a different approach
        // We'll create a ReadableStream that mimics OpenAI's SSE format
        const geminiClient = createGeminiClient(apiKey);
        
        // Extract system instruction if present
        let systemInstruction = '';
        const nonSystemMessages = [];
        
        for (const msg of messages) {
            if (msg.role === 'system') {
                systemInstruction = msg.content;
            } else {
                nonSystemMessages.push(msg);
            }
        }
        
        const chat = createGeminiChat(geminiClient, model || 'gemini-2.0-flash-exp', {
            temperature,
            maxOutputTokens: maxTokens || 8192,
            systemInstruction: systemInstruction || undefined
        });
        
        // Create a ReadableStream to handle Gemini's streaming
        const stream = new ReadableStream({
            async start(controller) {
                try {
                    console.log('[AIProviderService] Processing messages for Gemini:', nonSystemMessages.length, 'messages (excluding system)');
                    
                    // Get the last user message
                    const lastMessage = nonSystemMessages[nonSystemMessages.length - 1];
                    let lastUserMessage = lastMessage.content;
                    
                    // Handle case where content might be an array (multimodal)
                    if (Array.isArray(lastUserMessage)) {
                        // Extract text content from array
                        const textParts = lastUserMessage.filter(part => 
                            typeof part === 'string' || (part && part.type === 'text')
                        );
                        lastUserMessage = textParts.map(part => 
                            typeof part === 'string' ? part : part.text
                        ).join(' ');
                    }
                    
                    console.log('[AIProviderService] Sending message to Gemini:', 
                        typeof lastUserMessage === 'string' ? lastUserMessage.substring(0, 100) + '...' : 'multimodal content');
                    
                    // Prepare the message content for Gemini
                    let geminiContent = [];
                    
                    // Handle multimodal content properly
                    if (Array.isArray(lastMessage.content)) {
                        for (const part of lastMessage.content) {
                            if (typeof part === 'string') {
                                geminiContent.push(part);
                            } else if (part.type === 'text') {
                                geminiContent.push(part.text);
                            } else if (part.type === 'image_url' && part.image_url) {
                                // Convert base64 image to Gemini format
                                const base64Data = part.image_url.url.split(',')[1];
                                geminiContent.push({
                                    inlineData: {
                                        mimeType: 'image/png',
                                        data: base64Data
                                    }
                                });
                            }
                        }
                    } else {
                        geminiContent = [lastUserMessage];
                    }
                    
                    console.log('[AIProviderService] Prepared Gemini content:', 
                        geminiContent.length, 'parts');
                    
                    // Stream the response
                    let chunkCount = 0;
                    let totalContent = '';
                    
                    for await (const chunk of chat.sendMessageStream(geminiContent)) {
                        chunkCount++;
                        const chunkText = chunk.text || '';
                        totalContent += chunkText;
                        
                        // Format as SSE data
                        const data = JSON.stringify({
                            choices: [{
                                delta: {
                                    content: chunkText
                                }
                            }]
                        });
                        controller.enqueue(new TextEncoder().encode(`data: ${data}\n\n`));
                    }
                    
                    console.log(`[AIProviderService] Streamed ${chunkCount} chunks, total length: ${totalContent.length} chars`);
                    
                    // Send the final done message
                    controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
                    controller.close();
                    console.log('[AIProviderService] Gemini streaming completed successfully');
                } catch (error) {
                    console.error('[AIProviderService] Gemini streaming error:', error);
                    controller.error(error);
                }
            }
        });
        
        // Create a Response object with the stream
        return new Response(stream, {
            headers: {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        });
    } else if (provider === 'ollama') {
        console.log('[AIProviderService] Starting Ollama streaming request');
        const client = createOllamaClient();
        
        // Extract system instruction if present
        let systemInstruction = '';
        const nonSystemMessages = [];
        
        for (const msg of messages) {
            if (msg.role === 'system') {
                systemInstruction = msg.content;
            } else {
                nonSystemMessages.push(msg);
            }
        }
        
        // Create a prompt for Ollama
        const promptParts = [];
        for (const message of nonSystemMessages) {
            if (message.role === 'system') {
                promptParts.push(`System: ${message.content}`);
            } else if (message.role === 'user') {
                if (typeof message.content === 'string') {
                    promptParts.push(`User: ${message.content}`);
                } else if (Array.isArray(message.content)) {
                    // Handle multimodal content
                    for (const part of message.content) {
                        if (part.type === 'text') {
                            promptParts.push(`User: ${part.text}`);
                        } else if (part.type === 'image_url' && part.image_url?.url) {
                            // Extract base64 data from data URL
                            const base64Match = part.image_url.url.match(/^data:(.+);base64,(.+)$/);
                            if (base64Match) {
                                promptParts.push(`User sent an image`);
                            }
                        }
                    }
                }
            }
        }
        
        const prompt = promptParts.join('\n');
        console.log('[AIProviderService] Sending prompt to Ollama:', prompt);
        
        // Stream the response from Ollama
        const response = await client.streamGenerate(model || 'llama2', prompt);
        
        // Create a ReadableStream to handle Ollama's streaming
        const stream = new ReadableStream({
            async start(controller) {
                try {
                    console.log('[AIProviderService] Streaming response from Ollama');
                    
                    // Ollama streams in chunks
                    let totalContent = '';
                    response.body
                        .pipeThrough(new TextDecoderStream())
                        .pipeTo(new WritableStream({
                            write(chunk) {
                                console.log('[AIProviderService] Received chunk:', chunk);
                                totalContent += chunk;
                                
                                // Format as SSE data
                                const data = JSON.stringify({
                                    choices: [{
                                        delta: {
                                            content: chunk
                                        }
                                    }]
                                });
                                controller.enqueue(new TextEncoder().encode(`data: ${data}\n\n`));
                            },
                            close() {
                                console.log('[AIProviderService] Ollama streaming completed');
                                // Send the final done message
                                controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
                                controller.close();
                            },
                            abort(err) {
                                console.error('[AIProviderService] Ollama streaming error:', err);
                                controller.error(err);
                            }
                        }));
                } catch (error) {
                    console.error('[AIProviderService] Ollama streaming error:', error);
                    controller.error(error);
                }
            }
        });
        
        // Create a Response object with the stream
        return new Response(stream, {
            headers: {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        });
    } else {
        throw new Error(`Unsupported AI provider: ${provider}`);
    }
}

/**
 * Makes a streaming chat completion request with Portkey support
 * @param {object} params - Request parameters
 * @returns {Promise<Response>} The streaming response
 */
async function makeStreamingChatCompletionWithPortkey({ 
    apiKey, 
    provider = 'openai', 
    messages, 
    temperature = 0.7, 
    maxTokens = 1024, 
    model, 
    usePortkey = false,
    portkeyVirtualKey = null 
}) {
    if (!usePortkey) {
        return makeStreamingChatCompletion({ apiKey, provider, messages, temperature, maxTokens, model });
    }
    
    // Portkey is only supported for OpenAI currently
    if (provider !== 'openai') {
        console.warn('Portkey is only supported for OpenAI provider, falling back to direct API');
        return makeStreamingChatCompletion({ apiKey, provider, messages, temperature, maxTokens, model });
    }
    
    const fetchUrl = 'https://api.portkey.ai/v1/chat/completions';
    const response = await fetch(fetchUrl, {
        method: 'POST',
        headers: {
            'x-portkey-api-key': 'gRv2UGRMq6GGLJ8aVEB4e7adIewu',
            'x-portkey-virtual-key': portkeyVirtualKey || apiKey,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: model || 'gpt-4.1',
            messages,
            temperature,
            max_tokens: maxTokens,
            stream: true,
        }),
    });

    if (!response.ok) {
        throw new Error(`Portkey API error: ${response.status} ${response.statusText}`);
    }

    return response;
}

module.exports = {
    createAIClient,
    getGenerativeModel,
    makeChatCompletion,
    makeChatCompletionWithPortkey,
    makeStreamingChatCompletion,
    makeStreamingChatCompletionWithPortkey
};