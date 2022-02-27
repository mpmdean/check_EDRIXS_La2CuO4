FROM edrixs/edrixs_interactive@sha256:8c6f7199b9d210a84520b08fbd73577f528640ab8c1ffaa14d13bb108196b81d
RUN pip install lmfit==1.0.3 emcee==3.1.1
USER rixs
