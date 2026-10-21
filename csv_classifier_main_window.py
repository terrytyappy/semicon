




# -*- coding: utf-8 -*-
"""
csv_classifier_main_window.py
---------------------------------
GUIでCSVファイルを選択し、classify_rtn_core.py により
「RTN」または「nonRTN」に分類して出力するアプリ。

出力:
    元ファイル名_RTN.csv
    元ファイル名_RTN_summary.csv
"""
import os

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = (
    r"\\spe_server\common\実験結果\新垣\Shingaki"
    r"\.venv\Lib\site-packages\PyQt5\Qt5\plugins\platforms"
)
import os
import sys
import traceback
from typing import List
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QGridLayout,
    QListWidget, QPushButton, QLabel, QTextEdit, QStatusBar,
    QFileDialog, QMessageBox, QProgressBar
)

# 分類関数
from classify_rtn_core import classify_csv


# ---------- Helper UI Layout ----------
def create_vbox(widget_list, label_list=None, stretch=True, box_name='', direction='v'):
    box = QGroupBox(box_name)
    layout = QVBoxLayout() if direction == 'v' else QHBoxLayout()
    if label_list is None:
        label_list = [None] * len(widget_list)

    for wdc, widget in enumerate(widget_list):
        label = label_list[wdc]
        if label:
            row = QHBoxLayout()
            lbl = QLabel(label)
            row.addWidget(lbl)
            row.addWidget(widget)
            if not isinstance(widget, QTextEdit):
                row.addStretch(1)
            layout.addLayout(row)
        else:
            layout.addWidget(widget)
    if stretch:
        layout.addStretch(1)
    box.setLayout(layout)
    return box


def create_grid_box(widget_list, box_name=''):
    box = QGroupBox(box_name)
    layout = QGridLayout()
    for r, row in enumerate(widget_list):
        # ボタンやグループボックスが1つだけの行は横幅いっぱいに配置
        if len(row) == 1 and isinstance(row[0][0], (QPushButton, QGroupBox)):
            layout.addWidget(row[0][0], r, 0, 1, 99)
            continue
        for c, cell in enumerate(row):
            if len(cell) == 2:
                label, widget = cell
                hl = QHBoxLayout()
                if isinstance(label, str):
                    lab = QLabel(label)
                    hl.addWidget(lab)
                else:
                    hl.addWidget(label)
                hl.addWidget(widget)
                layout.addLayout(hl, r, c, 1, 1)
            else:
                layout.addWidget(cell[0], r, c, 1, 1)
    box.setLayout(layout)
    return box


# ---------- Main Window ----------
class CSVClassifierMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # スタイルは環境依存で落ちることがあるので安全に
        try:
            QApplication.setStyle('windowsvista')
        except Exception:
            pass

        self.setWindowTitle("CSV RTN Classifier")
        self.setMinimumSize(800, 600)

        # --- widgets ---
        # CSVファイル一覧
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(self.file_list.ExtendedSelection)

        # 出力パス
        self.output_path_edit = QTextEdit()
        self.output_path_edit.setFixedHeight(28)
        self.output_path_edit.setText(os.path.abspath("./output"))

        # ステータスバー
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 進捗バー
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        # --- build UI ---
        central = QWidget(self)
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        self._build_left_pane(main)

        # DnD
        self.setAcceptDrops(True)

    # ---------- Left Pane ----------
    def _build_left_pane(self, layout):
        # files
        add_btn = QPushButton("Add CSV Files")
        add_btn.clicked.connect(self._select_files)

        # フォルダ選択ボタン（サブフォルダなし）
        add_folder_btn = QPushButton("Add Folder (no subfolders)")
        add_folder_btn.clicked.connect(lambda: self._add_folder(recursive=False))

        # フォルダ選択ボタン（サブフォルダあり）
        add_folder_recursive_btn = QPushButton("Add Folder (with subfolders)")
        add_folder_recursive_btn.clicked.connect(lambda: self._add_folder(recursive=True))

        del_btn = QPushButton("Delete Selected")
        del_btn.clicked.connect(self._delete_selected)
        clr_btn = QPushButton("Clear All")
        clr_btn.clicked.connect(self.file_list.clear)

        file_box = create_vbox(
            [
                self.file_list,
                add_btn,
                add_folder_btn,
                add_folder_recursive_btn,
                del_btn,
                clr_btn
            ],
            [None] * 6,
            box_name="Select Files / Drag & Drop / Folders",
            stretch=False
        )

        # output
        out_btn = QPushButton("Select Output Folder")
        out_btn.clicked.connect(self._select_output_folder)
        out_box = create_grid_box(
            [
                [["Output Path", self.output_path_edit]],
                [[out_btn]]
            ],
            box_name="Output"
        )

        # run button
        run_group = QGroupBox("Run")
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Run RTN classification (percentile-based prominence, no height):"))
        run_btn = QPushButton("Run Classification")
        run_btn.clicked.connect(self._run_classification)
        hl.addWidget(run_btn)
        hl.addStretch(1)
        run_group.setLayout(hl)

        # assemble
        layout.addWidget(file_box)
        layout.addWidget(out_box)
        layout.addWidget(run_group)
        layout.addWidget(self.progress_bar)
        layout.addStretch(1)

    # ---------- File Operations ----------
    def _select_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select CSV Files", "", "CSV Files (*.csv)")
        for p in paths:
            if p and (p not in self._current_file_list()):
                self.file_list.addItem(p)

    def _add_folder(self, recursive: bool):
        """
        フォルダを選択し、その中のCSVをリストに追加する。
        recursive = False: そのフォルダ直下のCSVのみ
        recursive = True : サブフォルダも含めて全てのCSV
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing CSV Files", "")
        if not folder:
            return

        base = Path(folder)
        csv_files = sorted(base.rglob("*.csv")) if recursive else sorted(base.glob("*.csv"))

        if not csv_files:
            QMessageBox.information(self, "Info", "No CSV files found in the selected folder.")
            return

        existing = set(self._current_file_list())
        added = 0
        for p in csv_files:
            sp = str(p)
            if sp not in existing:
                self.file_list.addItem(sp)
                existing.add(sp)
                added += 1

        QMessageBox.information(
            self,
            "Info",
            f"Found {len(csv_files)} CSV files.\nAdded {added} new files to the list."
        )

    def _delete_selected(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

    def _select_output_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
        if d:
            self.output_path_edit.setText(d)

    def _current_file_list(self) -> List[str]:
        return [self.file_list.item(i).text() for i in range(self.file_list.count())]

    # ---------- Classification ----------
    def _run_classification(self):
        files = self._current_file_list()
        if not files:
            QMessageBox.warning(self, "Notice", "Please add at least one CSV file.")
            return

        outdir = self.output_path_edit.toPlainText().strip()
        if not outdir:
            QMessageBox.warning(self, "Notice", "Please select the output folder.")
            return

        os.makedirs(outdir, exist_ok=True)
        ok, fail = 0, 0
        total = len(files)

        # 進捗バー初期化
        self.progress_bar.setValue(0)

        for i, fp in enumerate(files, start=1):
            try:
                classify_csv(fp, outdir)
                ok += 1
                self.statusBar().showMessage(f"Done: {os.path.basename(fp)}")
            except Exception as e:
                fail += 1
                QMessageBox.critical(self, "Error", f"Failed on {fp}\n{e}")
                traceback.print_exc()

            # 進捗バー更新
            percent = int(i / total * 100)
            self.progress_bar.setValue(percent)
            QApplication.processEvents()

        QMessageBox.information(
            self, "Finished",
            f"Classification finished.\n"
            f"Success: {ok}\nFailed: {fail}\nOutput: {outdir}"
        )

    # ---------- Drag & Drop ----------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        existing = set(self._current_file_list())
        for p in paths:
            if p.lower().endswith(".csv") and (p not in existing):
                self.file_list.addItem(p)
                existing.add(p)


# ---------- Run App ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CSVClassifierMainWindow()
    w.show()
    sys.exit(app.exec_())
