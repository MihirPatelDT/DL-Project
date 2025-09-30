import tkinter as tk

# --- Main Application Class ---
class CalculatorApp:
    def __init__(self, root):
        """
        Initialize the calculator application.
        Sets up the main window, display, and buttons.
        """
        self.root = root
        self.root.title("Simple Calculator")
        self.root.geometry("400x600")
        self.root.resizable(False, False)
        self.root.configure(bg='#2c3e50') # Dark blue-grey background

        # The string to be displayed in the entry field
        self.expression = ""

        # --- UI Elements ---
        
        # Main frame to hold all widgets
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Display screen (Entry widget)
        display_frame = tk.Frame(main_frame, bg='#34495e', bd=10, relief=tk.SUNKEN)
        display_frame.pack(fill="x")
        
        self.display_var = tk.StringVar()
        self.display_entry = tk.Entry(display_frame, textvariable=self.display_var, 
                                      font=('Arial', 24, 'bold'), bd=0, 
                                      bg='#ecf0f1', fg='#2c3e50', justify='right')
        self.display_entry.pack(expand=True, fill='both', ipady=10)

        # Buttons frame
        buttons_frame = tk.Frame(main_frame, bg='#2c3e50')
        buttons_frame.pack(expand=True, fill="both", pady=10)

        self.create_buttons(buttons_frame)

    def create_buttons(self, parent_frame):
        """
        Creates and lays out all the calculator buttons.
        """
        # Button layout definition
        buttons = [
            ('7', 1, 0), ('8', 1, 1), ('9', 1, 2), ('/', 1, 3),
            ('4', 2, 0), ('5', 2, 1), ('6', 2, 2), ('*', 2, 3),
            ('1', 3, 0), ('2', 3, 1), ('3', 3, 2), ('-', 3, 3),
            ('0', 4, 0), ('.', 4, 1), ('+', 4, 2), ('=', 4, 3),
            ('C', 5, 0)
        ]

        # Style dictionaries for different button types
        num_style = {'bg': '#95a5a6', 'fg': '#2c3e50', 'font': ('Arial', 18, 'bold'), 'relief': tk.RAISED, 'bd': 5}
        op_style = {'bg': '#f39c12', 'fg': 'white', 'font': ('Arial', 18, 'bold'), 'relief': tk.RAISED, 'bd': 5}
        eq_style = {'bg': '#2ecc71', 'fg': 'white', 'font': ('Arial', 18, 'bold'), 'relief': tk.RAISED, 'bd': 5}
        clear_style = {'bg': '#e74c3c', 'fg': 'white', 'font': ('Arial', 18, 'bold'), 'relief': tk.RAISED, 'bd': 5}

        # Configure grid weights to make buttons expand
        for i in range(6):
            parent_frame.grid_rowconfigure(i, weight=1)
        for i in range(4):
            parent_frame.grid_columnconfigure(i, weight=1)
            
        # Create and place buttons in the grid
        for (text, row, col) in buttons:
            if text.isdigit() or text == '.':
                style = num_style
                action = lambda x=text: self.press(x)
            elif text in ['/', '*', '-', '+']:
                style = op_style
                action = lambda x=text: self.press(x)
            elif text == '=':
                style = eq_style
                action = self.calculate
            elif text == 'C':
                style = clear_style
                action = self.clear
            
            # Special handling for 'C' and '0' to span columns
            if text == 'C':
                button = tk.Button(parent_frame, text=text, command=action, **style)
                button.grid(row=row, column=col, columnspan=4, sticky="nsew", padx=5, pady=5)
            else:
                button = tk.Button(parent_frame, text=text, command=action, **style)
                button.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)


    def press(self, num):
        """
        Handles button presses for numbers and operators.
        Appends the pressed character to the expression.
        """
        self.expression += str(num)
        self.display_var.set(self.expression)

    def calculate(self):
        """
        Evaluates the current expression when '=' is pressed.
        Handles potential errors (e.g., division by zero).
        """
        try:
            # The eval function evaluates the string expression
            total = str(eval(self.expression))
            self.display_var.set(total)
            self.expression = total
        except:
            self.display_var.set("Error")
            self.expression = ""

    def clear(self):
        """
        Clears the display and resets the expression.
        """
        self.expression = ""
        self.display_var.set("")


# --- Main execution block ---
if __name__ == "__main__":
    root = tk.Tk()
    app = CalculatorApp(root)
    root.mainloop()