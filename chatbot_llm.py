import sys
import json
import requests
import markdown
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                             QLabel, QComboBox, QSplitter, QFrame, QToolBar,
                             QStatusBar, QAction, QFileDialog, QMessageBox,
                             QSystemTrayIcon, QMenu, QStyle, QSlider, QSpinBox,
                             QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QSettings
from PyQt5.QtGui import (QFont, QTextCursor, QPalette, QColor, QIcon, 
                         QTextCharFormat, QSyntaxHighlighter, QTextDocument,
                         QKeySequence, QPixmap)

class SyntaxHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for code blocks in markdown"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Define code formatting
        self.code_format = QTextCharFormat()
        self.code_format.setFontFamily("Courier")
        self.code_format.setBackground(QColor(40, 44, 52))
        self.code_format.setForeground(QColor(171, 178, 191))
        
    def highlightBlock(self, text):
        # Simple pattern for code blocks (```code```)
        pattern = r"```.*?```"
        
        import re
        for match in re.finditer(pattern, text, re.DOTALL):
            start = match.start()
            length = match.end() - match.start()
            self.setFormat(start, length, self.code_format)

class ChatWorker(QThread):
    """Worker thread for handling LLM requests without freezing the GUI"""
    response_signal = pyqtSignal(str, bool)  # response, is_complete
    error_signal = pyqtSignal(str)
    
    def __init__(self, message, history, model, temperature, max_tokens):
        super().__init__()
        self.message = message
        self.history = history
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.is_running = True

    def run(self):
        try:
            # Prepare the request to Ollama API
            url = "http://localhost:11434/api/chat"
            payload = {
                "model": self.model,
                "messages": self.history + [{"role": "user", "content": self.message}],
                "stream": True,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            response = requests.post(url, json=payload, stream=True)
            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if not self.is_running:
                        break
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.strip():
                            data = json.loads(decoded_line)
                            if 'message' in data and 'content' in data['message']:
                                content = data['message']['content']
                                full_response += content
                                self.response_signal.emit(content, False)
                            if data.get('done', False):
                                self.response_signal.emit("", True)
                                break
                if self.is_running:
                    # Add the final response to history
                    self.response_signal.emit(full_response, True)
            else:
                self.error_signal.emit(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            self.error_signal.emit(f"Connection error: {str(e)}")
    
    def stop(self):
        self.is_running = False

class ChatBubble(QFrame):
    """Custom widget for chat bubbles"""
    def __init__(self, message, is_user=False, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.setup_ui(message)
        
    def setup_ui(self, message):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Message label
        self.message_label = QLabel()
        self.message_label.setTextFormat(Qt.RichText)
        self.message_label.setWordWrap(True)
        self.message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        # Convert markdown to HTML
        html_content = markdown.markdown(message, extensions=['fenced_code'])
        self.message_label.setText(html_content)
        
        # Style based on sender
        if self.is_user:
            self.setStyleSheet("""
                ChatBubble {
                    background-color: #2a5aaa;
                    border-radius: 10px;
                    border-top-right-radius: 2px;
                    margin-left: 40px;
                }
                QLabel {
                    color: white;
                    background-color: transparent;
                }
            """)
            self.message_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        else:
            self.setStyleSheet("""
                ChatBubble {
                    background-color: #2d2d2d;
                    border-radius: 10px;
                    border-top-left-radius: 2px;
                    margin-right: 40px;
                }
                QLabel {
                    color: #e0e0e0;
                    background-color: transparent;
                }
            """)
            self.message_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        layout.addWidget(self.message_label)

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("MyCompany", "LLMChatbot")
        self.initUI()
        self.chat_history = []
        self.current_worker = None
        self.current_bubble = None
        self.setup_tray_icon()
        
    def initUI(self):
        self.setWindowTitle("LLM Chatbot Pro")
        self.setGeometry(100, 100, 1000, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for settings
        left_panel = QFrame()
        left_panel.setMaximumWidth(250)
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_panel)
        
        # Model selection
        model_label = QLabel("Model:")
        left_layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["llama2", "mistral", "codellama", "phi", "mixtral"])
        left_layout.addWidget(self.model_combo)
        
        # Temperature slider
        temp_label = QLabel("Temperature:")
        left_layout.addWidget(temp_label)
        
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 20)
        self.temp_slider.setValue(7)  # Default 0.7
        self.temp_slider.valueChanged.connect(self.update_temp_label)
        left_layout.addWidget(self.temp_slider)
        
        self.temp_value_label = QLabel("0.7")
        left_layout.addWidget(self.temp_value_label)
        
        # Max tokens
        tokens_label = QLabel("Max Tokens:")
        left_layout.addWidget(tokens_label)
        
        self.tokens_spin = QSpinBox()
        self.tokens_spin.setRange(100, 4000)
        self.tokens_spin.setValue(2000)
        left_layout.addWidget(self.tokens_spin)
        
        # Clear chat button
        self.clear_btn = QPushButton("Clear Chat")
        self.clear_btn.clicked.connect(self.clear_chat)
        left_layout.addWidget(self.clear_btn)
        
        left_layout.addStretch()
        
        # Right panel for chat
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Chat display area with scroll
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(10)
        self.chat_layout.setContentsMargins(10, 10, 10, 10)
        
        self.scroll_area.setWidget(self.chat_container)
        right_layout.addWidget(self.scroll_area)
        
        # Input area
        input_frame = QFrame()
        input_layout = QHBoxLayout(input_frame)
        
        self.input_field = QTextEdit()
        self.input_field.setMaximumHeight(80)
        self.input_field.setPlaceholderText("Type your message here...")
        input_layout.addWidget(self.input_field)
        
        self.send_button = QPushButton()
        self.send_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self.send_button.setFixedSize(40, 40)
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        right_layout.addWidget(input_frame)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # Create toolbar
        self.setup_toolbar()
        
        # Status bar
        self.statusBar().showMessage("Ready to chat")
        
        # Set style
        self.apply_dark_theme()
        
        # Load settings
        self.load_settings()
        
    def setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)
        
        # New chat action
        new_action = QAction(QIcon.fromTheme("document-new"), "New Chat", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self.new_chat)
        toolbar.addAction(new_action)
        
        # Save chat action
        save_action = QAction(QIcon.fromTheme("document-save"), "Save Chat", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_chat)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # Stop generation action
        self.stop_action = QAction(QIcon.fromTheme("process-stop"), "Stop Generation", self)
        self.stop_action.setShortcut(QKeySequence("Ctrl+."))
        self.stop_action.triggered.connect(self.stop_generation)
        self.stop_action.setEnabled(False)
        toolbar.addAction(self.stop_action)
        
    def setup_tray_icon(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
            
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        
        tray_menu = QMenu()
        show_action = tray_menu.addAction("Show")
        show_action.triggered.connect(self.show)
        
        hide_action = tray_menu.addAction("Hide")
        hide_action.triggered.connect(self.hide)
        
        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(QApplication.quit)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        self.tray_icon.activated.connect(self.tray_icon_activated)
        
    def tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.activateWindow()
        
    def update_temp_label(self, value):
        self.temp_value_label.setText(f"{value/10:.1f}")
        
    def apply_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(dark_palette)
        
        # Additional styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
            }
            QTextEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 5px;
                padding: 8px;
            }
            QComboBox, QSpinBox, QSlider {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 3px;
            }
            QPushButton {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QPushButton:pressed {
                background-color: #1d1d1d;
            }
            QLabel {
                color: #ffffff;
            }
        """)
    
    def send_message(self):
        message = self.input_field.toPlainText().strip()
        if not message:
            return
            
        # Add user message to chat
        self.add_message("user", message)
        self.input_field.clear()
        
        # Disable input while processing
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.stop_action.setEnabled(True)
        self.statusBar().showMessage("Thinking...")
        
        # Get current settings
        model = self.model_combo.currentText()
        temperature = self.temp_slider.value() / 10
        max_tokens = self.tokens_spin.value()
        
        # Create and start worker thread
        self.current_worker = ChatWorker(message, self.chat_history, model, temperature, max_tokens)
        self.current_worker.response_signal.connect(self.handle_response)
        self.current_worker.error_signal.connect(self.handle_error)
        self.current_worker.start()
    
    def handle_response(self, response, is_complete):
        if not hasattr(self, 'current_bubble') or self.current_bubble is None or is_complete:
            # Create new bubble for AI response
            self.current_bubble = ChatBubble(response, False)
            self.chat_layout.addWidget(self.current_bubble)
            
            if is_complete:
                # Add to history for context
                self.chat_history.append({"role": "assistant", "content": response})
                self.current_bubble = None
                
                # Re-enable input
                self.input_field.setEnabled(True)
                self.send_button.setEnabled(True)
                self.stop_action.setEnabled(False)
                self.statusBar().showMessage("Ready to chat")
                
                # Set focus back to input field
                self.input_field.setFocus()
                
                # Save settings
                self.save_settings()
        else:
            # Append to current bubble
            current_text = self.current_bubble.message_label.text()
            html_content = markdown.markdown(current_text + response, extensions=['fenced_code'])
            self.current_bubble.message_label.setText(html_content)
        
        # Scroll to bottom
        QTimer.singleShot(100, self.scroll_to_bottom)
    
    def scroll_to_bottom(self):
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def handle_error(self, error_message):
        self.add_message("system", error_message)
        
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.stop_action.setEnabled(False)
        self.statusBar().showMessage("Error occurred - check connection to Ollama")
        
        # Set focus back to input field
        self.input_field.setFocus()
    
    def add_message(self, sender, message):
        bubble = ChatBubble(message, sender == "user")
        self.chat_layout.addWidget(bubble)
        
        # Add to history for context (only user and assistant messages)
        if sender in ["user", "assistant"]:
            self.chat_history.append({"role": sender, "content": message})
        
        # Keep only the last 20 messages for context to avoid excessive memory usage
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
        
        # Scroll to bottom
        QTimer.singleShot(100, self.scroll_to_bottom)
    
    def stop_generation(self):
        if self.current_worker:
            self.current_worker.stop()
            self.current_worker = None
            
            # Re-enable input
            self.input_field.setEnabled(True)
            self.send_button.setEnabled(True)
            self.stop_action.setEnabled(False)
            self.statusBar().showMessage("Generation stopped")
    
    def clear_chat(self):
        # Clear the chat display
        for i in reversed(range(self.chat_layout.count())): 
            widget = self.chat_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        
        # Clear history
        self.chat_history = []
        
        # Add a welcome message
        self.add_message("system", "Chat cleared. Start a new conversation!")
    
    def new_chat(self):
        self.clear_chat()
    
    def save_chat(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Chat", "", "Text Files (*.txt);;JSON Files (*.json)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(self.chat_history, f, indent=2)
                else:
                    with open(file_path, 'w') as f:
                        for msg in self.chat_history:
                            f.write(f"{msg['role'].capitalize()}: {msg['content']}\n\n")
                
                self.statusBar().showMessage(f"Chat saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save chat: {str(e)}")
    
    def load_settings(self):
        # Load saved settings
        model = self.settings.value("model", "llama2")
        temperature = self.settings.value("temperature", 0.7, type=float)
        max_tokens = self.settings.value("max_tokens", 2000, type=int)
        
        # Apply settings
        index = self.model_combo.findText(model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        
        self.temp_slider.setValue(int(temperature * 10))
        self.tokens_spin.setValue(max_tokens)
    
    def save_settings(self):
        # Save current settings
        self.settings.setValue("model", self.model_combo.currentText())
        self.settings.setValue("temperature", self.temp_slider.value() / 10)
        self.settings.setValue("max_tokens", self.tokens_spin.value())
    
    def closeEvent(self, event):
        # Stop any ongoing generation
        if self.current_worker:
            self.current_worker.stop()
        
        # Save settings
        self.save_settings()
        
        # Hide to tray if minimized to tray
        if self.settings.value("minimize_to_tray", False, type=bool):
            event.ignore()
            self.hide()
            self.tray_icon.showMessage(
                "LLM Chatbot",
                "Application is still running in the system tray",
                QSystemTrayIcon.Information,
                2000
            )
        else:
            event.accept()

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("LLM Chatbot Pro")
    app.setApplicationVersion("1.0")
    
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()