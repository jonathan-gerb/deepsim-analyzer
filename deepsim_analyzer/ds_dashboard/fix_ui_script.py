from pathlib import Path
current_dir = Path(__file__).parents[0]
target_file = current_dir / "ui_form.py"
with open(str(target_file), "r") as file:
    filedata = file.read()

filedata = filedata.replace("PySide6", "PyQt6")
filedata = filedata.replace("sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)", "sizePolicy = QSizePolicy()")
filedata = filedata.replace("QFrame.HLine", "QFrame.Shape.HLine")
filedata = filedata.replace("QFrame.Sunken", "QFrame.Shadow.Sunken")
filedata = filedata.replace("slider.setOrientation(Qt.Horizontal)", "slider.setOrientation(Qt.Orientation.Horizontal)")

# Write the file out again
with open(str(target_file), 'w') as file:
  file.write(filedata)