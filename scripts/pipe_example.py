from brainsim.pipeline import run_pipes, mesh_pipes


pipe = mesh_pipes.CreateStl(rerun_all=True)
run_pipes(".", [pipe])