import tkinter as tk
from tkinter import ttk, messagebox
import torch, torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self): 
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): return self.net(x)

model = Net()
model.load_state_dict(torch.load("model.pth", map_location='cpu'))
model.eval()

means = [148.41, 719.5, 10.0, 0.1]
stds  = [260.54, 414.5, 10.0, 0.3]

def check():
    try:
        vals = [float(e.get() or 0) for e in [e1,e2,e3]]
        vals.append(1 if var.get() else 0)
        x = (np.array(vals) - means) / stds
        prob = torch.sigmoid(model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))).item()
        res = "FRAUD DETECTED!" if prob>0.5 else "Legitimate"
        color = "red" if prob>0.7 else "orange" if prob>0.3 else "green"
        lbl.config(text=f"{res}\nRisk: {prob*100:.1f}%", fg=color)
        canvas.delete("all")
        canvas.create_rectangle(10,10,10+prob*280,40,fill=color)
        canvas.create_text(150,25,text=f"{prob*100:.0f}%",fill="white",font=("Arial",12,"bold"))
        tree.insert("",0,values=(len(tree.get_children())+1, f"${vals[0]:,.0f}", f"{vals[1]} min", f"{vals[2]} km", "Yes" if vals[3] else "No", res))
    except: messagebox.showerror("Error","Enter valid numbers")

root = tk.Tk()
root.title("Bank Fraud Detector – Lab 16")
root.geometry("750x600")
root.configure(bg="#1e1e1e")

tk.Label(root,text="AI Real-time Fraud Detection System",font=("Arial",20,"bold"),fg="#00ff00",bg="#1e1e1e").pack(pady=20)

f = tk.Frame(root,bg="#1e1e1e")
tk.Label(f,text="Amount ($)",fg="white",bg="#1e1e1e").grid(row=0,column=0,pady=5)
e1 = tk.Entry(f); e1.grid(row=0,column=1)
tk.Label(f,text="Time since last (min)",fg="white",bg="#1e1e1e").grid(row=1,column=0,pady=5)
e2 = tk.Entry(f); e2.grid(row=1,column=1)
tk.Label(f,text="Location change (km)",fg="white",bg="#1e1e1e").grid(row=2,column=0,pady=5)
e3 = tk.Entry(f); e3.grid(row=2,column=1)
var = tk.BooleanVar()
tk.Checkbutton(f,text="International Transaction",variable=var,fg="white",bg="#1e1e1e").grid(row=3,column=0,columnspan=2,pady=10)
f.pack()

tk.Button(root,text="CHECK TRANSACTION",font=("Arial",14,"bold"),bg="#ff4444",fg="white",command=check).pack(pady=20)

lbl = tk.Label(root,text="Waiting for input...",font=("Arial",16),fg="white",bg="#1e1e1e")
lbl.pack(pady=10)

canvas = tk.Canvas(root,width=300,height=50,bg="#333333",highlightthickness=0)
canvas.pack(pady=10)
canvas.create_rectangle(10,10,10,40,fill="green")

tk.Label(root,text="Transaction History",font=("Arial",14),fg="#00ff00",bg="#1e1e1e").pack()
tree = ttk.Treeview(root,columns=("ID","Amt","Time","Loc","Intl","Result"),show="headings",height=10)
for c in tree["columns"]: tree.heading(c,text=c); tree.column(c,width=100)
tree.pack(padx=20,pady=10,fill="x")

root.mainloop()