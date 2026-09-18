"""Reading Guitar Pro scores, which is what people actually have lying around."""

import zipfile

import pytest

from tabify import TabifyError
from tabify.gpfile import read_gp, track_names

GPIF = """<GPIF>
 <Tracks>
  <Track id="0"><Name>Vocals</Name></Track>
  <Track id="1"><Name>Distortion Guitar</Name>
   <Staves><Staff><Properties>
     <Property name="Tuning"><Pitches>37 44 49 54 58 63</Pitches></Property>
   </Properties></Staff></Staves>
  </Track>
 </Tracks>
 <MasterTrack><Automations><Automation><Value>90 1</Value></Automation></Automations></MasterTrack>
 <MasterBars>
  <MasterBar><Bars>10 20</Bars></MasterBar>
  <MasterBar><Bars>11 21</Bars></MasterBar>
 </MasterBars>
 <Bars>
  <Bar id="10"><Voices>-1</Voices></Bar>
  <Bar id="20"><Voices>30</Voices></Bar>
  <Bar id="11"><Voices>-1</Voices></Bar>
  <Bar id="21"><Voices>31</Voices></Bar>
 </Bars>
 <Voices>
  <Voice id="30"><Beats>40 41</Beats></Voice>
  <Voice id="31"><Beats>42</Beats></Voice>
 </Voices>
 <Beats>
  <Beat id="40"><Notes>50 51 52</Notes></Beat>
  <Beat id="41"><Notes></Notes></Beat>
  <Beat id="42"><Notes>53</Notes></Beat>
 </Beats>
 <Notes>
  <Note id="50"><Properties>
    <Property name="String"><String>0</String></Property><Property name="Fret"><Fret>4</Fret></Property>
  </Properties></Note>
  <Note id="51"><Properties>
    <Property name="String"><String>1</String></Property><Property name="Fret"><Fret>4</Fret></Property>
  </Properties></Note>
  <Note id="52"><Properties>
    <Property name="String"><String>2</String></Property><Property name="Fret"><Fret>4</Fret></Property>
  </Properties></Note>
  <Note id="53"><Properties>
    <Property name="String"><String>0</String></Property><Property name="Fret"><Fret>0</Fret></Property>
  </Properties></Note>
 </Notes>
</GPIF>"""


@pytest.fixture
def score(tmp_path):
    path = tmp_path / "riff.gp"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("VERSION", "7.0")
        z.writestr("Content/score.gpif", GPIF)
    return path


def test_reads_the_guitar_track_not_the_first_one(score):
    tab = read_gp(score)
    assert tab.metadata["track"] == "Distortion Guitar"
    assert tab.tuning.strings == (37, 44, 49, 54, 58, 63)  # drop C#


def test_a_power_chord_is_one_stroke_not_three(score):
    # Three strings struck together are one stroke; the empty beat is a rest, not a stroke.
    assert read_gp(score).strokes[0] == [(0, 4), (1, 4), (2, 4)]


def test_bars_can_be_limited_to_what_was_actually_recorded(score):
    assert len(read_gp(score).strokes) == 2
    assert len(read_gp(score, bars=1).strokes) == 1


def test_reading_from_a_bar_in_the_middle_of_the_score(score):
    """A take is often one riff from deep inside a song, not the opening."""
    assert len(read_gp(score).strokes) == 2
    assert read_gp(score, from_bar=2).strokes == [[(0, 0)]]
    assert read_gp(score, from_bar=2, bars=1).strokes == [[(0, 0)]]
    assert read_gp(score, from_bar=1, bars=1).strokes == [[(0, 4), (1, 4), (2, 4)]]


def test_a_bar_past_the_end_reads_nothing_rather_than_wrapping(score):
    assert read_gp(score, from_bar=99).strokes == []


def test_picking_a_track_by_name(score):
    assert read_gp(score, track="vocals").metadata["track"] == "Vocals"
    with pytest.raises(TabifyError, match="no track matching"):
        read_gp(score, track="banjo")


def test_tempo_comes_across(score):
    assert read_gp(score).metadata["tempo"] == "90"


def test_track_names_lists_what_you_can_choose(score):
    assert track_names(score) == ["Vocals", "Distortion Guitar"]


def test_an_old_binary_gp5_says_so_rather_than_crashing(tmp_path):
    path = tmp_path / "old.gp"
    path.write_bytes(b"\x19FICHIER GUITAR PRO v5.10")
    with pytest.raises(TabifyError, match="Guitar Pro 7"):
        read_gp(path)
