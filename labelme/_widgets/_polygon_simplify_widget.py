from __future__ import annotations

from PySide6 import QtCore
from PySide6 import QtWidgets

from .. import _automation
from ._info_button import InfoButton


class PolygonSimplifyDialog(QtWidgets.QDialog):
    """Non-modal editor for the smoothing + RDP parameters.

    Edits take effect immediately via ``params_changed`` so the next AI
    proposal reflects them without closing the dialog.
    """

    params_changed = QtCore.Signal(object)  # _automation.PolygonSimplification
    apply_to_selected_requested = QtCore.Signal()

    def __init__(
        self,
        params: _automation.PolygonSimplification,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self.setWindowTitle(self.tr("Polygon Simplify Options"))

        self._epsilon = QtWidgets.QDoubleSpinBox()
        self._epsilon.setRange(0.1, 50.0)
        self._epsilon.setSingleStep(0.5)
        self._epsilon.setDecimals(1)
        self._epsilon.setSuffix(" px")
        self._epsilon.setToolTip(
            self.tr(
                "Ramer-Douglas-Peucker tolerance. Vertices closer than this to "
                "the simplified outline are dropped; larger means fewer points."
            )
        )

        self._iterations = QtWidgets.QSpinBox()
        self._iterations.setRange(0, 200)
        self._iterations.setToolTip(
            self.tr(
                "Number of smoothing passes before simplifying. Smoothing reaches "
                "about sqrt(iterations x (1 - w0)) pixels, e.g. 30 passes at "
                "w0 0.5 is about 4 px. 0 disables it."
            )
        )

        self._weight = QtWidgets.QDoubleSpinBox()
        self._weight.setRange(0.0, 1.0)
        self._weight.setSingleStep(0.05)
        self._weight.setDecimals(2)
        self._weight.setToolTip(
            self.tr(
                "Share each vertex keeps per pass; the rest comes from its two "
                "neighbors. Lower is smoother, 1.0 disables smoothing."
            )
        )

        self._merge_radius = QtWidgets.QSpinBox()
        self._merge_radius.setRange(0, 30)
        self._merge_radius.setSuffix(" px")
        self._merge_radius.setToolTip(
            self.tr(
                "Fuses mask pieces separated by gaps or thin necks up to about "
                "twice this width into one region. 0 disables merging."
            )
        )

        self._refine_method = QtWidgets.QComboBox()
        self._refine_method.addItem(self.tr("Off"), "none")
        self._refine_method.addItem(self.tr("Chan-Vese (intensity)"), "chan_vese")
        self._refine_method.addItem(self.tr("Geodesic (edges)"), "geodesic")
        self._refine_method.setToolTip(
            self.tr(
                "Level-set refinement snapping the boundary to the image. "
                "Chan-Vese suits bright objects such as implants; Geodesic follows "
                "strong edges when a touching neighbor has similar brightness."
            )
        )

        self._refine_iterations = QtWidgets.QSpinBox()
        self._refine_iterations.setRange(1, 100)
        self._refine_iterations.setToolTip(
            self.tr("Level-set iterations. More fixes larger errors but is slower.")
        )

        self._refine_band = QtWidgets.QSpinBox()
        self._refine_band.setRange(1, 30)
        self._refine_band.setSuffix(" px")
        self._refine_band.setToolTip(
            self.tr(
                "Max distance the refined boundary may move from the AI mask, "
                "keeping it from leaking into neighboring structures."
            )
        )

        self._snap_radius = QtWidgets.QDoubleSpinBox()
        self._snap_radius.setRange(0.0, 10.0)
        self._snap_radius.setSingleStep(0.5)
        self._snap_radius.setDecimals(1)
        self._snap_radius.setSuffix(" px")
        self._snap_radius.setToolTip(
            self.tr(
                "Final step: move each vertex along its normal onto the strongest "
                "image edge within this distance, so points sit on the boundary. "
                "0 disables snapping."
            )
        )

        form = QtWidgets.QFormLayout()
        form.addRow(self.tr("Merge fragments radius"), self._merge_radius)
        form.addRow(self.tr("Boundary refinement"), self._refine_method)
        form.addRow(self.tr("Refinement iterations"), self._refine_iterations)
        form.addRow(self.tr("Refinement band"), self._refine_band)
        form.addRow(self.tr("Smoothing iterations"), self._iterations)
        form.addRow(self.tr("Smoothing weight (w0)"), self._weight)
        form.addRow(self.tr("RDP epsilon"), self._epsilon)
        form.addRow(self.tr("Edge snap radius"), self._snap_radius)

        self._resegment = QtWidgets.QCheckBox(
            self.tr("Re-segment with SAM (box of each polygon)")
        )
        self._resegment.setToolTip(
            self.tr(
                "When applying to selected polygons, run the current AI model "
                "on each polygon's bounding box and replace the polygon with the "
                "post-processed SAM result. Manual edits to those polygons are "
                "replaced."
            )
        )

        apply_button = QtWidgets.QPushButton(self.tr("Apply to Selected Polygons"))
        apply_button.setToolTip(
            self.tr(
                "Run the selected polygons through the same merge, refinement, "
                "smoothing and simplification"
            )
        )
        apply_button.clicked.connect(self.apply_to_selected_requested)
        close_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        close_box.rejected.connect(self.close)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self._resegment)
        layout.addWidget(apply_button)
        layout.addWidget(close_box)
        self.setLayout(layout)

        self.set_params(params)
        for spin_box in (
            self._epsilon,
            self._iterations,
            self._weight,
            self._merge_radius,
            self._refine_iterations,
            self._refine_band,
            self._snap_radius,
        ):
            spin_box.valueChanged.connect(self._emit_params)
        self._refine_method.currentIndexChanged.connect(self._emit_params)
        self._resegment.toggled.connect(self._emit_params)
        self._refine_method.currentIndexChanged.connect(self._update_refine_enabled)
        self._update_refine_enabled()

    @property
    def params(self) -> _automation.PolygonSimplification:
        return _automation.PolygonSimplification(
            epsilon=self._epsilon.value(),
            smoothing_weight=self._weight.value(),
            smoothing_iterations=self._iterations.value(),
            merge_radius=self._merge_radius.value(),
            refine_method=self._refine_method.currentData(),
            refine_iterations=self._refine_iterations.value(),
            refine_band=self._refine_band.value(),
            snap_radius=self._snap_radius.value(),
            resegment_with_sam=self._resegment.isChecked(),
        )

    def set_params(self, params: _automation.PolygonSimplification) -> None:
        for editor, value in (
            (self._epsilon, params.epsilon),
            (self._iterations, params.smoothing_iterations),
            (self._weight, params.smoothing_weight),
            (self._merge_radius, params.merge_radius),
            (self._refine_iterations, params.refine_iterations),
            (self._refine_band, params.refine_band),
            (self._snap_radius, params.snap_radius),
        ):
            with QtCore.QSignalBlocker(editor):
                editor.setValue(value)
        with QtCore.QSignalBlocker(self._resegment):
            self._resegment.setChecked(params.resegment_with_sam)
        with QtCore.QSignalBlocker(self._refine_method):
            index = self._refine_method.findData(params.refine_method)
            self._refine_method.setCurrentIndex(max(index, 0))

    def _update_refine_enabled(self) -> None:
        enabled = self._refine_method.currentData() != "none"
        self._refine_iterations.setEnabled(enabled)
        self._refine_band.setEnabled(enabled)

    def _emit_params(self) -> None:
        self.params_changed.emit(self.params)


class PolygonSimplifyWidget(QtWidgets.QWidget):
    """Toolbar control: enable toggle plus a button opening the options dialog."""

    enabled_changed = QtCore.Signal(bool)
    options_requested = QtCore.Signal()

    def __init__(
        self, enabled: bool, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent=parent)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self.setLayout(layout)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.addStretch()
        header_layout.addWidget(QtWidgets.QLabel(self.tr("Polygon Simplify")))
        header_layout.addWidget(
            InfoButton(
                tooltip=self.tr(
                    "Post-process AI polygons: merge split mask pieces, snap the "
                    "boundary to the image with a level set, then smooth and "
                    "reduce points with Ramer-Douglas-Peucker"
                )
            )
        )
        header_layout.addStretch()
        layout.addLayout(header_layout)

        body_layout = QtWidgets.QHBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        self._checkbox = QtWidgets.QCheckBox(self.tr("Enable"))
        self._checkbox.setChecked(enabled)
        self._checkbox.toggled.connect(self.enabled_changed)
        body_layout.addWidget(self._checkbox)
        options_button = QtWidgets.QPushButton(self.tr("Options…"))
        options_button.clicked.connect(self.options_requested)
        body_layout.addWidget(options_button)
        layout.addLayout(body_layout)

    @property
    def is_enabled(self) -> bool:
        return self._checkbox.isChecked()

    def set_enabled_checked(self, enabled: bool) -> None:
        with QtCore.QSignalBlocker(self._checkbox):
            self._checkbox.setChecked(enabled)
