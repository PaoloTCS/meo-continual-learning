# Makefile for MEO Continual Learning Project

.PHONY: help ewc-grid update-tex pdf clean

# Default target
help:
	@echo "Available targets:"
	@echo "  ewc-grid    - Run EWC lambda grid search on MPS, then aggregate"
	@echo "  update-tex  - Update LaTeX with EWC results (requires EWC=<value>)"
	@echo "  pdf         - Build PDF from LaTeX source"
	@echo "  clean       - Clean build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make ewc-grid"
	@echo "  make update-tex EWC=63.4"
	@echo "  make pdf"

# Run EWC lambda grid search
ewc-grid:
	@chmod +x scripts/run_ewc_grid.sh
	@scripts/run_ewc_grid.sh

# Update LaTeX with EWC results
update-tex:
	@test $(EWC), "Usage: make update-tex EWC=63.4"
	@python scripts/update_tex_after_ewc.py --tex paper/MEO_Paper_v2_clean.tex --ewc $(EWC)

# Build PDF from LaTeX
pdf:
	cd paper && pdflatex MEO_Paper_v2_clean.tex

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@cd paper && rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex*
	@echo "Clean complete"
