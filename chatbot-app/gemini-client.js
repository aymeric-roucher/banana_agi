const { GoogleGenerativeAI } = require('@google/generative-ai');
const fs = require('fs');
const path = require('path');

class GeminiClient {
  constructor() {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error('GEMINI_API_KEY not found in environment variables');
    }

    this.genAI = new GoogleGenerativeAI(apiKey);
    this.model = this.genAI.getGenerativeModel({
      model: "gemini-2.5-flash-image-preview"
    });
  }

  /**
   * Convert image file to format expected by Gemini
   */
  async fileToGenerativePart(imagePath, mimeType) {
    return {
      inlineData: {
        data: fs.readFileSync(imagePath).toString('base64'),
        mimeType
      }
    };
  }

  /**
   * Get MIME type from file extension
   */
  getMimeType(filename) {
    const ext = path.extname(filename).toLowerCase();
    const mimeTypes = {
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.png': 'image/png',
      '.gif': 'image/gif',
      '.webp': 'image/webp'
    };
    return mimeTypes[ext] || 'image/jpeg';
  }

  /**
   * Build conversation context for Gemini
   */
  buildConversationContext(messages) {
    let context = "You are a helpful AI assistant. You can view and analyze images, and generate images when requested.\n\n";

    // Add previous messages as context (excluding the current one)
    if (messages.length > 1) {
      context += "Previous conversation:\n";
      for (let i = 0; i < messages.length - 1; i++) {
        const msg = messages[i];
        const role = msg.role === 'user' ? 'User' : 'Assistant';
        context += `${role}: ${msg.content}\n`;
        
        // Handle user uploaded image
        if (msg.image) {
          context += `${role} also shared an image.\n`;
        }
        
        // Handle assistant generated images
        if (msg.images && msg.images.length > 0) {
          const imageCount = msg.images.length;
          context += `${role} generated ${imageCount} image${imageCount > 1 ? 's' : ''}.\n`;
        }
      }
      context += "\nCurrent message:\n";
    }

    return context;
  }

  /**
   * Generate response using Gemini
   */
  async generateResponse(messages, imagePath = null) {
    try {
      const lastMessage = messages[messages.length - 1];
      const context = this.buildConversationContext(messages);

      // Prepare content parts
      const parts = [context + lastMessage.content];

      // Add recent images from conversation history for context
      // (Include images from last few messages so the model can reference them)
      const recentMessages = messages.slice(-3); // Last 3 messages
      for (const msg of recentMessages) {
        // Add user uploaded images
        if (msg.image && msg.role === 'user') {
          const userImagePath = path.join(__dirname, 'uploads', msg.image);
          if (fs.existsSync(userImagePath)) {
            const mimeType = this.getMimeType(userImagePath);
            const imagePart = await this.fileToGenerativePart(userImagePath, mimeType);
            parts.push(imagePart);
          }
        }
        
        // Add assistant generated images
        if (msg.images && msg.role === 'assistant') {
          for (const imageName of msg.images) {
            const assistantImagePath = path.join(__dirname, 'uploads', imageName);
            if (fs.existsSync(assistantImagePath)) {
              const mimeType = this.getMimeType(assistantImagePath);
              const imagePart = await this.fileToGenerativePart(assistantImagePath, mimeType);
              parts.push(imagePart);
            }
          }
        }
      }

      // Add current user image if provided (this takes precedence)
      if (imagePath && fs.existsSync(imagePath)) {
        const mimeType = this.getMimeType(imagePath);
        const imagePart = await this.fileToGenerativePart(imagePath, mimeType);
        parts.push(imagePart);
      }

      // Generate response
      const result = await this.model.generateContent(parts);
      const response = result.response;

      // Get text response if available
      let responseText = '';
      try {
        responseText = response.text();
      } catch (error) {
        // Some responses might not have text, only images
        responseText = '';
      }

      // Handle potential image generation response - based on Python implementation
      let generatedImages = [];

      // Extract images from response - following the Python pattern but collect ALL images
      if (response.candidates && response.candidates[0] && response.candidates[0].content.parts) {
        for (const part of response.candidates[0].content.parts) {
          if (part.inlineData && part.inlineData.data) {
            // Save generated image - similar to Python code
            const imageBuffer = Buffer.from(part.inlineData.data, 'base64');
            const filename = `generated-${Date.now()}-${generatedImages.length}.png`;
            const uploadsDir = path.join(__dirname, 'uploads');

            // Ensure uploads directory exists
            if (!fs.existsSync(uploadsDir)) {
              fs.mkdirSync(uploadsDir, { recursive: true });
            }

            const filepath = path.join(uploadsDir, filename);
            fs.writeFileSync(filepath, imageBuffer);
            generatedImages.push(filename);
          }
        }
      }

      // If we got images but no text, provide a default message
      if (generatedImages.length > 0 && !responseText) {
        responseText = `I've generated ${generatedImages.length} image${generatedImages.length > 1 ? 's' : ''} for you:`;
      }

      return {
        text: responseText,
        images: generatedImages
      };

    } catch (error) {
      console.error('Error generating response:', error);

      // Return a user-friendly error message
      if (error.message.includes('API key')) {
        throw new Error('Invalid API key. Please check your GEMINI_API_KEY environment variable.');
      } else if (error.message.includes('quota')) {
        throw new Error('API quota exceeded. Please try again later.');
      } else if (error.message.includes('safety')) {
        throw new Error('Content was blocked by safety filters. Please try rephrasing your message.');
      } else {
        throw new Error('Failed to generate response. Please try again.');
      }
    }
  }

  /**
   * Generate streaming response using Gemini
   */
  async generateStreamingResponse(messages, imagePath = null, onChunk = null) {
    try {
      const lastMessage = messages[messages.length - 1];
      const context = this.buildConversationContext(messages);
      
      // Prepare content parts
      const parts = [context + lastMessage.content];
      
      // Add image if provided
      if (imagePath && fs.existsSync(imagePath)) {
        const mimeType = this.getMimeType(imagePath);
        const imagePart = await this.fileToGenerativePart(imagePath, mimeType);
        parts.push(imagePart);
      }

      // Generate streaming response
      const result = await this.model.generateContentStream(parts);
      
      let responseText = '';
      let generatedImages = [];
      
      // Stream the response
      for await (const chunk of result.stream) {
        const chunkText = chunk.text();
        if (chunkText) {
          responseText += chunkText;
          // Call the callback with the incremental text
          if (onChunk) {
            onChunk({ type: 'text', content: chunkText, fullText: responseText });
          }
        }
        
        // Check for images in each chunk
        if (chunk.candidates && chunk.candidates[0] && chunk.candidates[0].content.parts) {
          for (const part of chunk.candidates[0].content.parts) {
            if (part.inlineData && part.inlineData.data) {
              const imageBuffer = Buffer.from(part.inlineData.data, 'base64');
              const filename = `generated-${Date.now()}-${generatedImages.length}.png`;
              const uploadsDir = path.join(__dirname, 'uploads');
              
              if (!fs.existsSync(uploadsDir)) {
                fs.mkdirSync(uploadsDir, { recursive: true });
              }
              
              const filepath = path.join(uploadsDir, filename);
              fs.writeFileSync(filepath, imageBuffer);
              generatedImages.push(filename);
              
              // Call the callback with the new image
              if (onChunk) {
                onChunk({ type: 'image', content: filename, allImages: [...generatedImages] });
              }
            }
          }
        }
      }

      // If we got images but no text, provide a default message
      if (generatedImages.length > 0 && !responseText) {
        responseText = `I've generated ${generatedImages.length} image${generatedImages.length > 1 ? 's' : ''} for you:`;
      }

      return {
        text: responseText,
        images: generatedImages
      };

    } catch (error) {
      console.error('Error generating streaming response:', error);
      
      // Return a user-friendly error message
      if (error.message.includes('API key')) {
        throw new Error('Invalid API key. Please check your GEMINI_API_KEY environment variable.');
      } else if (error.message.includes('quota')) {
        throw new Error('API quota exceeded. Please try again later.');
      } else if (error.message.includes('safety')) {
        throw new Error('Content was blocked by safety filters. Please try rephrasing your message.');
      } else {
        throw new Error('Failed to generate response. Please try again.');
      }
    }
  }

  /**
   * Generate response with conversation history support
   */
  async generateChatResponse(conversationHistory, newMessage, imagePath = null) {
    const messages = [...conversationHistory, newMessage];
    return this.generateResponse(messages, imagePath);
  }
}

module.exports = GeminiClient;