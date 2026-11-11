"""Config-driven spatial mapping reproduces the former per-model input_crop exactly."""
import json

import pytest

from worlddraft_eval.common import model_config
from worlddraft_eval.common.spatial_mapping import input_crop

# (model, output_size, expected crop or exception name) recorded from the pre-config implementation.
CASES = json.loads('''[["draft", [832, 480], [0, 0, 832, 480]], ["draft", [864, 480], [0, 0, 832, 480]], ["draft", [1920, 1080], [0, 0, 832, 480]], ["draft", [848, 480], [0, 0, 832, 480]], ["draft", [832, 492], [0, 0, 832, 480]], ["draft", [480, 832], [0, 0, 832, 480]], ["draft", [1280, 720], [0, 0, 832, 480]], ["MiniMax", [832, 480], [0, 0, 832, 480]], ["MiniMax", [864, 480], [0, 0, 832, 480]], ["MiniMax", [1920, 1080], [0, 0, 832, 480]], ["MiniMax", [848, 480], [0, 0, 832, 480]], ["MiniMax", [832, 492], [0, 0, 832, 480]], ["MiniMax", [480, 832], [0, 0, 832, 480]], ["MiniMax", [1280, 720], [0, 0, 832, 480]], ["OmniWeaving", [832, 480], [0.0, 0.0, 832.0, 480.0]], ["OmniWeaving", [864, 480], [5.684341886080802e-14, 8.888888888888914, 831.9999999999999, 462.2222222222222]], ["OmniWeaving", [1920, 1080], [-5.684341886080802e-14, 5.999999999999972, 832.0000000000001, 468.00000000000006]], ["OmniWeaving", [848, 480], [0.0, 4.528301886792434, 832.0, 470.94339622641513]], ["OmniWeaving", [832, 492], [10.146341463414615, -2.842170943040401e-14, 811.7073170731708, 480.00000000000006]], ["OmniWeaving", [480, 832], [277.53846153846155, 0.0, 276.9230769230769, 480.0]], ["OmniWeaving", [1280, 720], [0.0, 6.0, 832.0, 468.0]], ["UniVideo", [832, 480], [0, 0, 832.0, 480.0]], ["UniVideo", [864, 480], "ValueError"], ["UniVideo", [1920, 1080], "ValueError"], ["UniVideo", [848, 480], "ValueError"], ["UniVideo", [832, 492], "ValueError"], ["UniVideo", [480, 832], "ValueError"], ["UniVideo", [1280, 720], "ValueError"], ["VINO", [832, 480], [0, 0, 824.0761904761905, 475.0515463917526]], ["VINO", [864, 480], "ValueError"], ["VINO", [1920, 1080], "ValueError"], ["VINO", [848, 480], "ValueError"], ["VINO", [832, 492], "ValueError"], ["VINO", [480, 832], "ValueError"], ["VINO", [1280, 720], "ValueError"], ["Wan2.1-VACE-14B", [832, 480], [0, 0, 832, 480]], ["Wan2.1-VACE-14B", [864, 480], [0, 0, 832, 480]], ["Wan2.1-VACE-14B", [1920, 1080], [0, 0, 832, 480]], ["Wan2.1-VACE-14B", [848, 480], [0, 0, 832, 480]], ["Wan2.1-VACE-14B", [832, 492], [0, 0, 832, 480]], ["Wan2.1-VACE-14B", [480, 832], [0, 0, 832, 480]], ["Wan2.1-VACE-14B", [1280, 720], [0, 0, 832, 480]], ["LynnReal-Omni", [832, 480], [0, 0, 832, 480]], ["LynnReal-Omni", [864, 480], [0, 0, 832, 480]], ["LynnReal-Omni", [1920, 1080], [0, 0, 832, 480]], ["LynnReal-Omni", [848, 480], [0, 0, 832, 480]], ["LynnReal-Omni", [832, 492], [0, 0, 832, 480]], ["LynnReal-Omni", [480, 832], [0, 0, 832, 480]], ["LynnReal-Omni", [1280, 720], [0, 0, 832, 480]], ["Bernini-R-14B", [832, 480], [0, 0, 832, 480]], ["Bernini-R-14B", [864, 480], [0, 0, 832, 480]], ["Bernini-R-14B", [1920, 1080], [0, 0, 832, 480]], ["Bernini-R-14B", [848, 480], [0, 0, 832, 480]], ["Bernini-R-14B", [832, 492], [0, 0, 832, 480]], ["Bernini-R-14B", [480, 832], [0, 0, 832, 480]], ["Bernini-R-14B", [1280, 720], [0, 0, 832, 480]], ["Wan-3.0", [832, 480], [0, 0, 832, 480]], ["Wan-3.0", [864, 480], [0, 0, 832, 480]], ["Wan-3.0", [1920, 1080], [0, 0, 832, 480]], ["Wan-3.0", [848, 480], [0, 0, 832, 480]], ["Wan-3.0", [832, 492], [0, 0, 832, 480]], ["Wan-3.0", [480, 832], [0, 0, 832, 480]], ["Wan-3.0", [1280, 720], [0, 0, 832, 480]], ["SolarRenderer", [832, 480], [0.0, 0.0, 832.0, 480.0]], ["SolarRenderer", [864, 480], [-16.0, 0.0, 864.0, 480.0]], ["SolarRenderer", [1920, 1080], [-10.666666666666666, 0.0, 853.3333333333334, 480.0]], ["SolarRenderer", [848, 480], [-8.0, 0.0, 848.0, 480.0]], ["SolarRenderer", [832, 492], [0.0, -6.0, 832.0, 492.0]], ["SolarRenderer", [480, 832], [0.0, -480.0, 832.0, 1441.7328519855596]], ["SolarRenderer", [1280, 720], [-10.666666666666666, 0.0, 853.3333333333334, 480.0]], ["Wan2.2-VACE-Fun-A14B", [832, 480], [0, 0, 832, 480]], ["Wan2.2-VACE-Fun-A14B", [864, 480], "ValueError"], ["Wan2.2-VACE-Fun-A14B", [1920, 1080], "ValueError"], ["Wan2.2-VACE-Fun-A14B", [848, 480], "ValueError"], ["Wan2.2-VACE-Fun-A14B", [832, 492], "ValueError"], ["Wan2.2-VACE-Fun-A14B", [480, 832], "ValueError"], ["Wan2.2-VACE-Fun-A14B", [1280, 720], "ValueError"], ["C2R", [832, 480], [0, 0, 832, 480]], ["C2R", [864, 480], "ValueError"], ["C2R", [1920, 1080], "ValueError"], ["C2R", [848, 480], "ValueError"], ["C2R", [832, 492], "ValueError"], ["C2R", [480, 832], "ValueError"], ["C2R", [1280, 720], "ValueError"], ["Seedance-2.5", [832, 480], [0, -6, 832, 492]], ["Seedance-2.5", [864, 480], [0, -6, 832, 492]], ["Seedance-2.5", [1920, 1080], [0, -6, 832, 492]], ["Seedance-2.5", [848, 480], [0, -6, 832, 492]], ["Seedance-2.5", [832, 492], [0, -6, 832, 492]], ["Seedance-2.5", [480, 832], [0, -6, 832, 492]], ["Seedance-2.5", [1280, 720], [0, -6, 832, 492]]]''')


@pytest.mark.parametrize('model,output,expected', CASES)
def test_mapping_matches_former_implementation(model, output, expected):
    if isinstance(expected, str):
        with pytest.raises(ValueError if expected == 'ValueError' else Exception):
            input_crop(model, (832, 480), tuple(output))
    else:
        assert input_crop(model, (832, 480), tuple(output)) == expected


def test_unknown_model_names_the_config_file():
    with pytest.raises(KeyError, match='configs/models.yaml'):
        input_crop('NewModel', (832, 480), (832, 480))


def test_reference_is_draft_p2_aa():
    assert model_config.reference() == 'draft' and model_config.is_camera_reference('draft', 'P2_AA')
    assert not model_config.is_camera_reference('MiniMax', 'P2_AA')
