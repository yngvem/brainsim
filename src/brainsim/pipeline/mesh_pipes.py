import meshio
from .pipe import Pipe
from subprocess import run


class CreateStl(Pipe):
    required_glob_patterns = [
        'surf/lh.pial',
        'surf/rh.pial',
    ]

    def run_pipe(self, directory):
        lh_pial = directory / "surf/lh.pial"
        rh_pial = directory / "surf/rh.pial"
        
        stl_dir = directory / "stl"
        lh_pial_stl = stl_dir / "lh.pial.stl"
        rh_pial_stl = stl_dir / "rh.pial.stl"
        stl_dir.mkdir(exist_ok=True)
        
        print(f"mris_convert {lh_pial} {lh_pial_stl}")
        run(["mris_convert", str(lh_pial), str(lh_pial_stl)])
        print(f"mris_convert {rh_pial} {rh_pial_stl}")
        run(["mris_convert", str(rh_pial), str(rh_pial_stl)])

    def check_already_ran(self, directory):
        if not (directory / "stl/lh.pial.stl").is_file():
            return False
        if not (directory / "stl/rh.pial.stl").is_file():
            return False
        
        return True
            