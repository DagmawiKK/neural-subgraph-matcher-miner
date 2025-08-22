import { useState, useEffect, useRef } from 'react';

const useWebSocket = (url, options = {}) => {
  const [socket, setSocket] = useState(null);
  const [lastMessage, setLastMessage] = useState(null);
  const [readyState, setReadyState] = useState(0);
  const [connected, setConnected] = useState(false);
  
  const reconnectTimeoutId = useRef(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = options.maxReconnectAttempts || 5;
  const reconnectInterval = options.reconnectInterval || 3000;

  const connect = () => {
    try {
      const ws = new WebSocket(url);
      
      ws.onopen = (event) => {
        console.log('WebSocket connected');
        setReadyState(ws.readyState);
        setConnected(true);
        reconnectAttempts.current = 0;
        
        if (options.onOpen) {
          options.onOpen(event);
        }
      };
      
      ws.onmessage = (event) => {
        const message = event;
        setLastMessage(message);
        
        if (options.onMessage) {
          options.onMessage(message);
        }
      };
      
      ws.onclose = (event) => {
        console.log('WebSocket disconnected');
        setReadyState(ws.readyState);
        setConnected(false);
        
        if (options.onClose) {
          options.onClose(event);
        }
        
        // Attempt to reconnect
        if (reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++;
          console.log(`Attempting to reconnect... (${reconnectAttempts.current}/${maxReconnectAttempts})`);
          
          reconnectTimeoutId.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };
      
      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        
        if (options.onError) {
          options.onError(event);
        }
      };
      
      setSocket(ws);
      setReadyState(ws.readyState);
      
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
    }
  };

  useEffect(() => {
    connect();
    
    return () => {
      if (reconnectTimeoutId.current) {
        clearTimeout(reconnectTimeoutId.current);
      }
      
      if (socket) {
        socket.close();
      }
    };
  }, [url]);

  const sendMessage = (message) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(message);
    } else {
      console.warn('WebSocket is not connected');
    }
  };

  return {
    socket,
    lastMessage,
    readyState,
    connected,
    sendMessage
  };
};

export default useWebSocket;