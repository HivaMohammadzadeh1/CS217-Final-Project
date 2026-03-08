# Quick Setup

This file is intentionally short.

For the current project status, start with:
- [/Users/dannyadkins/CS217-Final-Project/README.md](/Users/dannyadkins/CS217-Final-Project/README.md)
- [/Users/dannyadkins/CS217-Final-Project/docs/CURRENT_STATE_AND_SWEEP_PLAN.md](/Users/dannyadkins/CS217-Final-Project/docs/CURRENT_STATE_AND_SWEEP_PLAN.md)

## Local sanity checks

```bash
make fpga-doctor
make fpga-ref-sim
make py-tests
make -C systemc run
```

## If you are working on FPGA deployment

Read:
- [/Users/dannyadkins/CS217-Final-Project/fpga/README.md](/Users/dannyadkins/CS217-Final-Project/fpga/README.md)
- [/Users/dannyadkins/CS217-Final-Project/integration/README_LAB1.md](/Users/dannyadkins/CS217-Final-Project/integration/README_LAB1.md)

## Note

Earlier versions of this repo had several long setup guides. This file is now just a quick entry point so the README remains the main source of truth.
