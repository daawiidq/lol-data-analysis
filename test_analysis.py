import tempfile
from pathlib import Path
import unittest
import pandas as pd
from analysis import load_matches, objective_summary

class CleaningTests(unittest.TestCase):
    def write(self, rows):
        tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        p=Path(tmp.name)/'matches.csv'; pd.DataFrame(rows).to_csv(p,index=False); return p
    def row(self, game=1, **changes):
        r=dict(gameId=game,gameDuration=1200,winner=1,firstBlood=1,firstDragon=0,firstBaron=2,extra=10)
        r.update(changes); return r
    def test_exact_duplicates_removed(self):
        data=load_matches(self.write([self.row(),self.row(),self.row(2)]))
        self.assertEqual(len(data),2); self.assertEqual(data.attrs['audit']['exact_duplicates_removed'],1)
    def test_conflicts_in_unused_columns_rejected(self):
        with self.assertRaises(ValueError):load_matches(self.write([self.row(),self.row(extra=11)]))
    def test_invalid_duration_excluded(self):
        data=load_matches(self.write([self.row(),self.row(2,gameDuration=-1)]))
        self.assertEqual(len(data),1)
    def test_no_event_excluded(self):
        data=load_matches(self.write([self.row(),self.row(2,firstBlood=0)]))
        self.assertEqual(objective_summary(data,'firstBlood')['games'],1)
if __name__=='__main__':unittest.main()
