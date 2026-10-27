#!/usr/bin/env python3
"""Synthetic SWED registry and hospital-EHR extracts.

Writes four linked Parquet files that mimic the real SWED extracts:

    synth_ed_core.parquet            emergency department visits
    synth_patreg_wide.parquet        national patient register, one row per episode
    synth_deathreg_wide.parquet      cause of death register, one row per deceased person
    synth_danderyd_lab_meas.parquet  vital signs and lab results (Danderyd and SÖS)

Every target comes from "Synthetic SWED Parquet files: LLM instructions"
(Oct 13, 2026), which was itself written from a column profile of the real
files. Values the profile did not contain are marked ASSUMPTION and kept as
named constants below.

    uv run --with numpy --with pyarrow python make_synthetic_swed.py --n-persons 100000
    uv run --with numpy --with pyarrow python make_synthetic_swed.py --n-persons 100000 --validate
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# =============================================================================
# Dates are int64 days since 1970-01-01 throughout; times are minutes or µs.
# =============================================================================

EPOCH = np.datetime64("1970-01-01", "D")


def D(s: str) -> int:
    return int((np.datetime64(s, "D") - EPOCH).astype(np.int64))


def ymd(days):
    d = np.asarray(days, dtype=np.int64).astype("datetime64[D]")
    months = d.astype("datetime64[M]")
    mi = months.astype(np.int64)
    return mi // 12 + 1970, mi % 12 + 1, (d - months.astype("datetime64[D]")).astype(np.int64) + 1


def from_ymd(y, m, d):
    months = (np.asarray(y, np.int64) - 1970) * 12 + np.asarray(m, np.int64) - 1
    start = months.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64)
    length = (months + 1).astype("datetime64[M]").astype("datetime64[D]").astype(np.int64) - start
    return start + np.minimum(np.asarray(d, np.int64), length) - 1


def add_years(days, k):
    y, m, d = ymd(days)
    return from_ymd(y + k, m, d)


def age_years(birth, at):
    by, bm, bd = ymd(birth)
    ay, am, ad = ymd(at)
    return ay - by - ((am * 100 + ad) < (bm * 100 + bd)).astype(np.int64)


def iso(day) -> str:
    return str(EPOCH + np.timedelta64(int(day), "D"))


# =============================================================================
# Constants from the spec (sections 2-7)
# =============================================================================

REAL_PERSONS = 2_172_844
REAL_ROWS = dict(ed=7_801_285, patreg=61_649_322, death=275_954, lab=16_599_550)
ROWS_PER_PERSON = {k: v / REAL_PERSONS for k, v in REAL_ROWS.items()}
FILES = dict(
    ed="synth_ed_core.parquet",
    patreg="synth_patreg_wide.parquet",
    death="synth_deathreg_wide.parquet",
    lab="synth_danderyd_lab_meas.parquet",
)

ED_START, ED_END = D("2003-04-02"), D("2017-12-18")
REG_START, REG_END = D("1996-01-01"), D("2016-12-31")
OUTPAT_START = D("2001-01-01")
UTDATUM_MIN = D("1996-10-15")
DEATH_END = D("2017-12-31")
LAB_START, LAB_END = D("2005-01-01"), D("2017-08-31")
LAB_MIN_MINUTE = 14 * 60  # real min is 2005-01-01 14:00
NO_DEATH = D("2200-01-01")

P_MALE = 0.482
P_FOREIGN = 0.241
P_DEATH = 0.127
ED_MALE_ROW_SHARE = 0.459
PATREG_FEMALE_ROW_SHARE = 0.579
INPAT_SHARE = 0.128

# =============================================================================
# ASSUMPTIONS (the profile had no number for these)
# =============================================================================

ID_MAX = 9_999_999
P_BIRTH_JAN1 = 0.01
# Birth-date tail knots below q05: linear interpolation from 1895 to 1924 would
# put 2% of persons into birth years 1895-1908 (age 95+ at the first ED visit).
BIRTH_KNOTS = [(0.0, "1895-01-01"), (0.002, "1905-07-01"), (0.015, "1915-07-01"), (0.05, "1924-07-01"),
               (0.25, "1946-07-01"), (0.50, "1964-07-01"), (0.75, "1980-07-01"), (0.95, "1992-07-01"),
               (1.0, "1999-11-06")]
ED_DATE_KNOTS = [(0.0, "2003-04-02"), (0.05, "2005-10-20"), (0.25, "2009-07-14"), (0.5, "2012-09-07"),
                 (0.75, "2015-03-25"), (0.95, "2017-03-10"), (1.0, "2017-12-18")]
HOME_HOSPITAL_SHARE = 0.90
ROWS_PER_VISIT_P = np.array([0.88, 0.10, 0.02])  # 1, 2, 3 rows
ED_NB_K = 0.7
ED_AGE_SLOPE = 0.020  # log visit rate per year of age
ED_Z_SLOPE = 0.5
DEATH_AGE_COEF = 0.11  # logit per year of age in 2010
DEATH_Z_COEF = 0.5
P_NEAR_DEATH_VISIT = 0.36  # calibrated so ~40% of deceased have an ED visit in their last 30 days
OBDUK_ROW_SHARE = 0.00012
UTSATT4_SHARE = 0.014
OUTPAT_FROM_ED = 0.8
LOS_BINS = [(0, 0, 0.10), (1, 1, 0.15), (2, 2, 0.13), (3, 3, 0.11), (4, 7, 0.28), (8, 14, 0.14),
            (15, 30, 0.06), (31, 120, 0.03)]
DIA_ANT_NULL = 0.576  # DIA_ANT is present on the latest 42.4% of episodes (about 2011-04 on)
OP_ANT_SHARE = 0.61
EKOD_ST_SHARE = 0.85
SENINV_YYYYMM_SHARE = 0.30
SENINV_PLACEHOLDER = 0.023
SENUTV_PLACEHOLDER = 0.15
SENUTV_1965_SHARE = 0.03
DEATH_INEXACT = 0.008
DEATH_YEAR_ONLY = 0.6  # of inexact deaths; the rest lack only the day
ULORSAK_NULL = 0.102
CANCER_ECHO = 0.70
HF_ECHO = 0.60
ERR_ED_AFTER_DEATH = 0.001
ERR_EPISODE_AFTER_DEATH = 0.0005
ERR_LAB_DISPLACED = 0.01
LAB_DS_WEIGHT = 1.6  # relative selection weight of Danderyd vs SÖS visits
LAB_GARBAGE_RATE = 1e-5
PLACEHOLDER_SPACES = "       ."

# =============================================================================
# Category lists from the spec, verbatim
# =============================================================================

LIST_A = """value,rows
buksmärta [21522001],566230
bröstsmärta [29857009],501022
har svårt att andas [230145002],259819
svullnad av extremitet [80068009]+smärta i extremitet [90834002],191075
yrsel [404640003],189428
akut buk [9209005],179310
huvudskada [82271004],164490
huvudvärk [25064002],162748
onormal hjärtrytm [361136003],150218
dyspné [267036007],140182
ryggsmärta [161891005],136739
sårskada [125643001],135968
infektionssjukdom [40733004],113565
feber [386661006],113378
fotskada [125604000],100523
urinvägsbesvär [59421000052104],96253
handskada [125599006],91346
förgiftning [75478009],87315
fynd som rör buken [609624008],85230
hjärtfynd [301095005],80106
skada i höftregionen [125600009],78605
allergiskt tillstånd [473011001],77559
knäskada [125601008],75046
graviditetsproblem [289209003],71773
synkope [271594007],70999
skada på övre extremitet [127278005],67362
neurologiskt bortfallssymtom [264552009],64190
smärta i extremitet [90834002],63317
sjukdomskänsla [367391008],60260
fynd som rör vaginal blödning [289530006],58435
"krampanfall, obestämt om fokalt eller generaliserat [277130004]",58193
skada i axel och överarm [699062006],53341
flanksmärta [247355005],51713
ont i halsen [267102003],47286
traumatisk skada [417746004],43728
diarré [62315008],40724
handledsskada [125598003],40642
traumatisk och/eller icke-traumatisk ryggskada [712893003],39934
skada på ankel [125603006],39176
blödning från näsan [249366005],38410
"""

LIST_B = """value,rows,hospital
MED,675813,danderyd
KIR,562907,södersjukhuset
AKUTEN,493208,sahlgrenska
ORT,405144,danderyd
AKMCKÖ,390936,sahlgrenska
AKUTMS,376609,sahlgrenska
Akutmottagning Västerås,306101,västerås
S - Akutflöde,260710,karolinska
H - Akutmott (Med 1),246937,karolinska
"Medicinkliniken, CStG",245472,st_göran
GYN,230249,södersjukhuset
KARD,212794,södersjukhuset
H - Akutmott (Kir 1),200080,karolinska
Vuxen,166604,hosp_07
"Kirurgkliniken, CStG",139909,st_göran
S - Akutmott (ÖNH),136402,karolinska
H - Akutmott (Ort 1),132867,karolinska
"Ortopedkliniken, CStG",128422,st_göran
-,114361,södersjukhuset
Akutmott Medicin,110773,hosp_08
HJÄ,103716,södersjukhuset
Akutmott Kirurg,96065,hosp_09
H - Akutmott (Neuro 1),90867,karolinska
S - Akutmott (Ort),88388,karolinska
H - Akutmott (Gyn),88214,karolinska
S - Akutmott (Neurol),68772,karolinska
H - Akutmott (Inf),66874,karolinska
S - Akutmott (Inf),64489,karolinska
SSK,57815,södersjukhuset
"""

LIST_C = """value,rows
Gående,3612268
Ambulans,1070716
Larm med ambulans,80660
Larm Ambulans,47091
Sjuktransport,17324
Färdtjänst,17188
Bårtaxi,15117
Polis,14252
Bår,10193
Larm ambulans övrigt,8539
Helikopter,8398
Liggande sjuktransport,4696
Inkommer med polis,4089
Internt,3533
Taxi,3141
Konsult (inneliggande SÖS),3105
Larm ambulans Trauma B,2983
Larm Gående,2874
Ambulans TR3,2594
Ambulans TR2,2343
Ambulans TR1,2249
Larm ambulans Rädda hjärnan,1513
Hänvisad från 1177,1337
Larm med ambulans TR2,1314
Larm med ambulans TR1,1311
Larm med ambulans TR3,1284
Larm ambulans,953
Larm Helikopter,797
Larm med helikopter,777
Helikopter TR2,574
Helikopter TR1,494
Larm ambulans Trauma A,395
Larm med helikopter TR2,274
Helikopter TR3,202
Larm Polis,200
Larm med helikopter TR1,196
Kriminalvård,185
Larm gående övrigt,148
Larm gående Trauma B,143
Larm ambulans Trauma Nivå 2,108
"""

LIST_D = """value,rows
Hem,3051844
Intagen,359469
30 - Från/Till hemmet,255752
hem,169453
39 - Till slutenvård,121793
Avvikit,87960
Registrering saknas,64555
Annan inrättning,41246
Inlagda sös,41100
Annat sjukhus,39214
Hemmet,36159
S - AVA 1,27015
H - MAVA M81 Medicin,25574
S - AVA 2,25509
Hänvisad,24598
Inskriven avd 36,20707
H - MAVA M83 Medicin,19989
Annat,18414
DS Hjärt avd 94,18354
30 Utskriven till hem,17606
Inskriven avd 33,15326
DS Kirurgi avd 63,14909
Inskriven avd 50,14715
33 - PV-bokad på VC/JC,13767
DS Medicin 73 Stroke,13571
Annan vårdenhet,13446
Direkt inskrivning,13402
H - Neurolog avd R82,13141
STSAB - Med avd 10,12854
Avviker,11003
S - R15 Neurologi,10929
H - KAVA K43 Kirurg/ort,10760
39 Direkt inskrivning,10640
Inskriven avd 35,10334
DS Medicin 75 IMA,10327
DS Gynekologi avd 14,9838
Satellit,9604
STSAB - Med avd 8,9414
H - KAVA K41 Kirurg,9362
Inskriven avd 32S,9072
H - Kvinnoklin Gynavd K67,8997
S - N25 HIA,8501
H - Ortoped avd K83,8465
Inskriven avd 31,8210
DS Kirurgi avd 61,8014
"""

LIST_E = """value,rows
Hem,93778
DS Medicin,57660
DS Hjärt,45121
DS Kirurgi,37574
IntmedAVA,32576
51,26287
52,23417
Avdelning 4,23107
31,19495
Avdelning 3,16186
36,15351
72,14809
DS Ortopedi,14333
Obsenheten,13995
DALEN,13928
23,13863
81,13791
83,11978
Avvikit,11457
57,11336
34,10366
56,10081
24,9531
Avdelning 2,9421
67,8988
Avdelning 1,8894
25,8314
33,7966
27,7859
KirAVA,7572
"""

LIST_F = """value,rows
Buksmärta,511708
Bröstsmärta,318009
Andningsbesvär,202361
AKUTBUK,163385
Huvudvärk,118232
Yrsel,117784
Extremitet svullnad/värk,115251
BRSM,104432
Feber,101360
Huvudskada,95875
Rytmrubbning,77963
Graviditetsrelaterade besvär,69990
Ryggvärk,69302
DYSPNE,66704
Diagnosinformation saknas,64943
Urinvägsbesvär,61589
Sårskada,61205
Sjukdomskänsla,59275
Höftskada,58536
Vaginal blödning,58028
Fotskada,56457
Knäskada,55130
HJÄRTA,53102
Axelskada,52254
INF,51531
TRHAND,48628
Halsont,47079
Flanksmärta,47069
YRSEL,46783
BUK,46044
Svimning,44517
TRHUVUD,43543
Förgiftning,42923
Handledsskada,40610
BORTFALL,40408
"Extremitet svullnad, värk",39485
TRIAGE,39328
Fotledsskada,39147
Buk,37614
Handskada,37147
Allergisk reaktion,37073
Bröstsmärtor,36063
TRFOT,35578
Ospecificerad kontaktorsak,34111
EXTSMÄRT,33728
"""

LIST_G = """value,rows
R10-R19,1323327
Z51,1057046
R00-R09,1012797
R50-R69,985488
Z09,983616
Z49,882517
H25-H28,722020
D10-D36,704755
Z03,651075
H30-H36,634665
H40-H42,621099
F10,611789
M79,606845
G40-G47,585559
Z01,537528
O80-O84,501448
Z08,498824
F41,494797
D37-D48,452071
M05,450750
F32,432538
M54,430794
C81-C96,428930
Z71,387466
L80-L99,386418
O20-O29,364669
C60-C63,364010
L20-L30,363954
M17,345589
H15-H22,344073
F31,343073
N39,336526
I489,336501
S52,330492
C76-C80,326264
C50-C50,310603
F33,310545
C15-C26,309181
R40-R46,308763
F90,298905
"""

LIST_H = """value,rows
I109,765571
N18,664554
C76-C80,392982
C50-C50,325952
Z85,318164
H30-H36,248717
C60-C63,227027
Z09,217544
F10,213553
O60-O75,208152
F41,197105
E119,195753
I489,190495
H25-H28,179754
R50-R69,170858
Z95,152633
D10-D36,151764
I509,140336
R10-R19,139508
Z96,135431
C15-C26,132744
G40-G47,126580
J44,118797
F32,117370
H40-H42,116931
"""

LIST_I = """value,rows
I109,235177
Z95,103468
E119,86888
Z92,74033
I489,72081
C76-C80,65155
O60-O75,62250
I509,59976
I252,58751
E785,50931
Z85,47208
E039,46780
"""

LIST_J = """value,rows
W0199,498312
W1999,415164
X5999,310309
W0109,108669
Y883,97083
Y831,79969
W2299,75733
Y838,73992
Y839,63909
Y869,57225
W1099,54779
X5899,54474
W0099,53250
Y579,45012
Y0799,35456
X6499,35344
X5099,34807
W4499,33661
W2699,32627
W0130,31463
"""

LIST_K = """value,rows
XS012,872910
AU009,811658
AL003,609849
DV071,518760
DR016,494793
XS100,373436
AN032,326990
AF034,315241
DR014,284800
XS900,283059
DT016,280960
AC026,280200
CJE20,271992
DT026,268558
AV034,249562
UJD02,231990
AU118,229015
UKC02,194109
AC022,177797
AF022,174638
AC041,154254
AQ002,150621
AF021,147696
AF063,145091
XS915,143669
"""

LIST_L = """value,rows
ZV031,514145
AU118,188709
AV034,186060
DT026,175453
AU009,173012
XS915,137692
AC034,133468
XV008,119809
DT016,115063
AU010,113618
ZXH10,98976
ZXH60,98448
AL003,94734
DV071,90729
AV018,86479
"""

LIST_M = """value,rows
51001,6884642
11001,6717892
11002,5463597
10013,4876212
11010,4297061
10011,3016257
56010,2236712
51014,1986267
10012,1350434
11011,1293767
11012,829734
12001,823506
11014,804890
10492,705723
10481,641180
10483,630512
10484,600075
52012,487794
50071,456432
11335,449328
57010,440210
50070,369365
56012,344458
10XXXX,344225
51013,314026
12010,305798
10601,273847
10505,245223
21001,240521
55010,228480
10468,221459
57013,218499
53014,216057
13010,213495
11013,210364
51012,208674
10493,197932
41001,195066
"""

LIST_N = """value,rows
I219,16175
C349,11273
I251,7128
C61,6566
F03,6062
I259,5144
G309,4987
J449,4977
C259,4970
I64,4824
I509,4794
J189,4790
C509,4490
C189,3960
I639,3601
I48,3532
I258,3229
F039,3226
R54,3171
R999,3024
I110,2815
F019,2667
I619,2623
X590,2310
I709,2298
I5099,2262
I489,2225
A419,2200
I694,2170
J440,2074
C20,2014
W19,2010
C679,1889
C56,1737
I350,1683
E149,1676
G20,1619
C719,1578
C169,1526
I10,1460
"""

LIST_O = """value,rows
I469,19291
R579,15742
J189,12093
I509,11930
I219,10059
C349,8431
R688,8308
R54,6014
I5099,4632
A419,4423
C61,4304
I5090,4284
J969,4030
C259,3945
J81,3076
C509,2998
F03,2927
R53,2910
R068,2739
C189,2708
"""

LIST_P = """value,rows
I509,8734
I709,8390
J189,6235
C798,6087
J449,5587
I251,5498
I219,5348
I259,5117
I10,5050
R54,4381
C787,4310
I5099,3754
I48,3612
C799,3389
R53,3164
"""

LIST_Q = """value,rows
I10,6121
E149,3637
J449,3507
I709,3032
I509,2954
I48,2852
N19,2207
R54,2057
F03,2008
I489,1884
I5099,1876
J189,1803
"""

LIST_R = """value,rows
T71,1191
S720,876
S7200,871
T402,563
T404,495
T751,488
S065,464
S0650,450
T07,421
T401,331
T510,302
S069,301
S068,257
S7210,242
S721,229
T403,226
S0680,211
T432,203
T179,189
S7290,178
T58,177
T436,165
T426,159
T430,152
T149,144
T433,142
S729,127
T424,123
T300,117
T509,115
T818,108
"""

LIST_S = """value,rows
0180,81982
1480,36842
0188,7089
0184,7049
0126,6789
0182,6751
1980,6445
0160,5872
0127,5725
0136,5668
0181,5405
0123,5329
0186,5297
0163,5163
1490,5111
1481,4198
0162,3640
0183,3585
0117,3185
0114,3083
0138,3000
0191,2936
0192,2885
"""

LIST_T_SHARES = {  # term_name: real rows
    "Blodtryck systoliskt - övre": 1940185, "Blodtryck diastoliskt - nedre": 1817792, "Saturation": 1788903,
    "Andningsfrekvens": 1649918, "Kroppstemperatur": 1530195, "Pulsfrekvens": 1020984,
    "Medvetandegrad enl Retts": 431877, "B-Leukocyter": 392361, "B-Hemoglobin": 391849, "B-Trombocyter": 390353,
    "B-EVF": 388192, "P-Kreatinin": 381428, "P-CRP": 380951, "P-Kalium": 339148, "P-Natrium": 337911,
    "P-Glukos": 282348,
}

LIST_V = """value,rows
mmol/L,1725332
x10(9)/L,800283
g/L,525069
mikromol/L,498791
mg/L,492853
mikrokat/L,255332
nanog/L,140542
kPa,102564
%,98211
s,44857
L/min,40002
ng/L,27661
mm,6699
mikrog/L,5427
pmol/L,2027
antal/synf,384
"""

LIST_W = """value,rows
137-145,439769
"4,0-6,0",436377
"3,5-8,8",392320
<3,380990
"3,5-4,6",339146
117-153,211528
165-387,210331
"0,35-0,46",209448
<90,206536
<100,182004
134-170,180320
145-348,180020
"0,39-0,50",178706
<15,140542
"<0,76",110401
<26,107168
"3,6-4,6",101811
"0,5-2,3",100910
22-27,100322
<8,88836
8-16,85137
"5,3-6,6",61650
"7,32-7,43",61320
-3-3,57012
120-150,54350
"<0,61",54047
"<1,5",49089
"<2,0",49084
98-107,48516
130-160,45727
"4,6-6,0",40914
"7,35-7,45",40126
28-40,38308
"<1,1",36557
"<3,5",26553
"<0,25",22026
<222,13006
<194,12712
"<4,3",9786
34-45,9322
"""


def parse_list(text):
    rows = list(csv.reader(io.StringIO(text.strip())))
    body = rows[1:]
    return [r[0] for r in body], np.array([float(r[1]) for r in body]), [r[2:] for r in body]


# =============================================================================
# Generic sampling helpers
# =============================================================================

def knots(pairs):
    p = np.array([a for a, _ in pairs], float)
    v = np.array([D(b) if isinstance(b, str) else b for _, b in pairs], float)
    return p, v


def inv_cdf(u, kn):
    return np.interp(u, kn[0], kn[1])


def cdf(x, kn):
    return np.interp(x, kn[1], kn[0])


def qsample(rng, n, kn, lo=None, hi=None):
    """Inverse-CDF sample from quantile knots, optionally truncated to [lo, hi]."""
    u = rng.random(n)
    a = cdf(lo, kn) if lo is not None else 0.0
    b = cdf(hi, kn) if hi is not None else 1.0
    return inv_cdf(a + u * (b - a), kn)


def qsample_days(rng, n, kn, lo, hi):
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    x = np.floor(qsample(rng, n, kn, lo, hi + 0.999)).astype(np.int64)
    return np.clip(x, lo.astype(np.int64), hi.astype(np.int64))


def bisect(f, lo, hi, it=80):
    flo = f(lo)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def capped_zipf(k, total, cap, s=1.0):
    """k weights summing to total, Zipf-shaped, none above cap."""
    if k == 0 or total <= 0:
        return np.zeros(k)
    w = 1.0 / np.arange(1, k + 1) ** s
    w = w / w.sum() * total
    for _ in range(200):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = w < cap
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w


def with_tail(rng, values, counts, real_rows, nonnull_share, pool, s=1.0):
    """Listed values (real counts) plus a capped Zipf tail from pool filling the uncovered share.

    Returns (values, weights) with weights as shares of all rows."""
    listed = list(values)
    w = np.asarray(counts, float) / real_rows
    seen = set(listed)
    tail = [t for t in pool if t not in seen]
    tail = [tail[i] for i in rng.permutation(len(tail))]
    tw = capped_zipf(len(tail), nonnull_share - w.sum(), w.min() * 0.9, s)
    return listed + tail, np.concatenate([w, tw])


def choice_p(rng, n, w):
    w = np.asarray(w, float)
    return rng.choice(len(w), size=n, p=w / w.sum())


def sample_grouped(rng, group, weights, allowed):
    """Sample value indices per row. allowed is (G, V) bool; weights are target marginal shares.

    Values allowed only in some groups get boosted there so the marginal is kept."""
    n = len(group)
    G = allowed.shape[0]
    frac = np.bincount(group, minlength=G) / max(n, 1)
    reach = frac @ allowed
    w = np.where(reach > 0, np.asarray(weights, float) / np.maximum(reach, 1e-12), 0.0)
    out = np.zeros(n, np.int64)
    for g in range(G):
        rows = np.flatnonzero(group == g)
        if len(rows) == 0:
            continue
        p = w * allowed[g]
        out[rows] = rng.choice(len(p), size=len(rows), p=p / p.sum())
    return out


def allocate(rng, out, key, item_keys, item_ids, targets):
    """Give item j up to targets[j] still-free rows whose key equals item_keys[j]."""
    item_keys = np.asarray(item_keys)
    for g in np.unique(item_keys):
        sel = np.flatnonzero(item_keys == g)
        rows = np.flatnonzero((key == g) & (out < 0))
        rng.shuffle(rows)
        start = 0
        for j in sel:
            take = int(min(targets[j], len(rows) - start))
            if take <= 0:
                continue
            out[rows[start:start + take]] = item_ids[j]
            start += take
    return out


def quantile_map(rng, days, kn, lo, hi):
    """Move days so their empirical distribution follows kn, keeping their order; clip to [lo, hi]."""
    u = (np.argsort(np.argsort(days + rng.random(len(days)), kind="stable")) + 0.5) / max(len(days), 1)
    return np.clip(np.floor(inv_cdf(u, kn)).astype(np.int64), lo, hi)


def exact_draw(rng, n, probs):
    """Categories in exactly the given proportions, in random order."""
    return rank_assign(rng.random(n), counts_from_shares(n, probs))


def draw_nb(rng, mu, k):
    mu = np.asarray(mu, float)
    out = np.zeros(len(mu), np.int64)
    pos = mu > 0
    out[pos] = rng.negative_binomial(k, k / (k + mu[pos]))
    return out


def draw_zt_nb(rng, mu, k):
    p = k / (k + mu)
    out = rng.negative_binomial(k, p)
    for _ in range(200):
        zero = np.flatnonzero(out == 0)
        if len(zero) == 0:
            break
        out[zero] = rng.negative_binomial(k, p[zero])
    out[out == 0] = 1
    return out


def draw_los(rng, n):
    probs = np.array([b[2] for b in LOS_BINS])
    b = choice_p(rng, n, probs)
    lo = np.array([x[0] for x in LOS_BINS])[b]
    hi = np.array([x[1] for x in LOS_BINS])[b]
    return lo + np.floor(rng.random(n) * (hi - lo + 1)).astype(np.int64)


def rank_assign(score, counts):
    """Assign category c to counts[c] rows in ascending order of score."""
    order = np.argsort(score, kind="stable")
    out = np.empty(len(score), np.int64)
    edges = np.concatenate([[0], np.cumsum(counts)])
    edges[-1] = len(score)
    for c in range(len(counts)):
        out[order[edges[c]:edges[c + 1]]] = c
    return out


def counts_from_shares(n, shares):
    shares = np.asarray(shares, float)
    c = np.floor(n * shares / shares.sum()).astype(np.int64)
    c[np.argmax(shares)] += n - c.sum()
    return c


def vocab_index(values):
    return {v: i for i, v in enumerate(values)}


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


# =============================================================================
# Hospitals
# =============================================================================

HOSPITALS = ["karolinska", "södersjukhuset", "sahlgrenska", "danderyd", "st_göran", "västerås",
             "hosp_07", "hosp_08", "hosp_09", "hosp_10", "hosp_11", "hosp_12"]
KS, SOS, SU, DS, STG, VAS = range(6)
HOSP_SHARE = np.array([22.64, 17.73, 16.16, 14.39, 11.94, 3.92] + [2.20] * 6)
HOSP_P = HOSP_SHARE / HOSP_SHARE.sum()
ABBREV_STYLE = np.array([h in (KS, STG) for h in range(12)])
# ED-linked register episodes use a fixed SJUKHUS code per ED hospital (assumption).
HOSP_SJUKHUS = [["11001", "11002"], ["10013"], ["51001"], ["11010"], ["11012"], ["56010"],
                ["12001"], ["21001"], ["41001"], ["55010"], ["57010"], ["53014"]]
HOSP_LK = [
    "0114 0115 0117 0120 0123 0125 0126 0127 0128 0136 0138 0139 0140 0160 0162 0163 0180 0181 0182 0183 0184 0186 0187 0188 0191 0192",
    None, "1401 1402 1407 1415 1419 1421 1427 1430 1435 1438 1439 1440 1441 1442 1443 1444 1445 1446 1447 1452 1460 1461 1462 1463 1465 1466 1470 1471 1472 1473 1480 1481 1482 1484 1485 1486 1487 1488 1489 1490 1491 1492 1493 1494 1495 1496 1497 1498 1499",
    None, None, "1904 1907 1960 1961 1962 1980 1981 1982 1983 1984",
    "0305 0319 0330 0331 0360 0380 0381 0382",
    "0509 0512 0513 0560 0561 0562 0563 0580 0581 0582 0583 0584 0586",
    "1214 1230 1231 1233 1256 1257 1260 1261 1262 1263 1264 1265 1266 1267 1270 1272 1273 1275 1276 1277 1278 1280 1281 1282 1283 1284 1285 1286 1287 1290 1291 1292 1293",
    "1814 1860 1861 1862 1863 1864 1880 1881 1882 1883 1884 1885",
    "2260 2262 2280 2281 2282 2283 2284",
    "2401 2403 2404 2409 2417 2418 2421 2422 2425 2460 2462 2463 2480 2481 2482",
]
HOSP_REGION = [0, 0, 2, 0, 0, 5, 6, 7, 8, 9, 10, 11]  # Stockholm hospitals share one region

RARE_CARE_UNITS = {
    KS: ["S - Akutmott (Kir)", "H - Akutmott (Ögon)", "S - Akutmott (Hand)", "H - Akutmott (Psyk)",
         "S - Akutmott (Gyn)", "H - Akutmott (ÖNH)", "S - Barnakuten", "H - Akutmott (Uro)", "H - Akutmott (Med 2)"],
    STG: ["Akutkliniken, CStG", "Geriatrik, CStG", "Infektion, CStG", "Psykiatri, CStG", "Närakut, CStG"],
    DS: ["NEU", "URO", "INF", "ÖGON"],
    SOS: ["ÖNH", "HAND", "PSYK", "INF-S"],
    VAS: ["Akutmottagning Västerås Barn", "Jourcentral Västerås"],
    6: ["Vuxen Obs", "Barn"],
    7: ["Akutmott Ortopedi", "Akutmott Infektion", "Akutmott Gyn", "Akutmott Obs"],
    8: ["Akutmott Medicin S", "Akutmott Ortopedi S", "Akutmott Barn S", "Akutmott Obs S"],
    **{h: [f"AKM {u} {h + 1:02d}" for u in ["Medicin", "Kirurgi", "Ortopedi", "Infektion", "Obs",
                                            "Gyn", "Psyk", "Närakut", "Barn"]] for h in (9, 10, 11)},
}
CARE_UNIT_OUTSIDE_NULL = {STG: 0.9, KS: 0.8}

# =============================================================================
# Presenting complaints (list A plus an assumed tail)
# =============================================================================

# term -> (concept group, urgent, sex rule, register dx key)
COMPLAINT_INFO = {
    "buksmärta": ("abd", 0, None, "abd"), "bröstsmärta": ("chest", 1, None, "chest"),
    "har svårt att andas": ("breath", 1, None, "breath"),
    "svullnad av extremitet": ("ext", 0, None, "ext"), "yrsel": ("dizzy", 0, None, "neuro"),
    "akut buk": ("abd", 0, None, "abd"), "huvudskada": ("headinj", 0, None, "inj:S06,S00,S01"),
    "huvudvärk": ("headache", 0, None, "neuro"), "onormal hjärtrytm": ("arrhythmia", 1, None, "arrhythmia"),
    "dyspné": ("breath", 1, None, "breath"), "ryggsmärta": ("back", 0, None, "back"),
    "sårskada": ("wound", 0, None, "inj:S01,S61,S81,S51,S41,S91"), "infektionssjukdom": ("fever", 0, None, "fever"),
    "feber": ("fever", 0, None, "fever"), "fotskada": ("foot", 0, None, "inj:S92,S93,S90"),
    "urinvägsbesvär": ("urinary", 0, None, "uti"), "handskada": ("hand", 0, None, "inj:S62,S63,S60,S61"),
    "förgiftning": ("poison", 0, None, "poison"), "fynd som rör buken": ("abd", 0, None, "abd"),
    "hjärtfynd": ("chest", 0, None, "chest"), "skada i höftregionen": ("hip", 0, None, "inj:S72,S70,S73"),
    "allergiskt tillstånd": ("allergy", 0, None, "allergy"), "knäskada": ("knee", 0, None, "inj:S83,S80,S82"),
    "graviditetsproblem": ("preg", 0, "F1850", "preg"), "synkope": ("syncope", 0, None, "neuro"),
    "skada på övre extremitet": ("upper", 0, None, "inj:S42,S52,S40,S50"),
    "neurologiskt bortfallssymtom": ("deficit", 1, None, "stroke"), "smärta i extremitet": ("ext", 0, None, "ext"),
    "sjukdomskänsla": ("malaise", 0, None, "malaise"),
    "fynd som rör vaginal blödning": ("vag", 0, "F1850", "vag"),
    "krampanfall, obestämt om fokalt eller generaliserat": ("seizure", 0, None, "seizure"),
    "skada i axel och överarm": ("shoulder", 0, None, "inj:S42,S43,S40"),
    "flanksmärta": ("flank", 0, None, "flank"), "ont i halsen": ("throat", 0, None, "throat"),
    "traumatisk skada": ("trauma", 0, None, "inj:T14,S00,S20,S30"), "diarré": ("diarr", 0, None, "diarr"),
    "handledsskada": ("wrist", 0, None, "inj:S52,S62,S63"),
    "traumatisk och/eller icke-traumatisk ryggskada": ("backinj", 0, None, "inj:S32,S33,S22"),
    "skada på ankel": ("ankle", 0, None, "inj:S82,S93"), "blödning från näsan": ("epistaxis", 0, None, "epistaxis"),
}

# ASSUMPTION: tail terms in the same `term [SNOMED CT id]` format.
COMPLAINT_TAIL = [
    ("cerebrovaskulär händelse", "230690007", "stroke", 1, None, "stroke"),
    ("sänkt medvetande", "3006004", "consc", 1, None, "neuro"),
    ("illamående", "422587007", "abd", 0, None, "abd"),
    ("kräkning", "422400008", "abd", 0, None, "abd"),
    ("hosta", "49727002", "breath", 0, None, "breath"),
    ("utslag", "271807003", "skin", 0, None, "skin"),
    ("ögonbesvär", "41652007", "eye", 0, None, "eye"),
    ("öronvärk", "16001004", "ear", 0, None, "ear"),
    ("psykisk störning", "74732009", "psych", 0, None, "psych"),
    ("självskada", "276853009", "psych", 0, None, "poison"),
    ("alkoholförgiftning", "25702006", "psych", 0, None, "alcohol"),
    ("förstoppning", "14760008", "abd", 0, None, "abd"),
    ("hematuri", "34436003", "urinary", 0, None, "uti"),
    ("urinretention", "267064002", "urinary", 0, None, "uti"),
    ("rektal blödning", "12063002", "abd", 0, None, "gi_bleed"),
    ("hematemes", "8765009", "abd", 1, None, "gi_bleed"),
    ("trötthet", "84229001", "malaise", 0, None, "malaise"),
    ("ångest", "48694002", "psych", 0, None, "psych"),
    ("förvirring", "286933003", "consc", 0, None, "neuro"),
    ("fall", "1912002", "trauma", 0, None, "inj:S00,S72,S52,S32"),
    ("hypoglykemi", "302866003", "diab", 1, None, "diab"),
    ("hyperglykemi", "80394007", "diab", 0, None, "diab"),
    ("högt blodtryck", "38341003", "htn", 0, None, "htn"),
    ("främmande kropp", "125670008", "fb", 0, None, "fb"),
    ("brännskada", "125666000", "burn", 0, None, "inj:T20,T21,T22,T23,T24,T30"),
    ("djurbett", "399907009", "wound", 0, None, "inj:S61,S81,S51"),
    ("tandvärk", "27355003", "dental", 0, None, "dental"),
    ("synstörning", "63102001", "eye", 0, None, "eye"),
    ("domningar", "44077006", "deficit", 0, None, "neuro"),
    ("svaghet", "13791008", "malaise", 0, None, "malaise"),
    ("revbensfraktur", "33737001", "trauma", 0, None, "inj:S22"),
    ("skada i ansiktet", "125592005", "headinj", 0, None, "inj:S02,S01"),
    ("nackskada", "262530004", "backinj", 0, None, "inj:S13,S12"),
    ("hudinfektion", "108365000", "skin", 0, None, "skin_inf"),
    ("abscess", "44132006", "skin", 0, None, "skin_inf"),
    ("smärta i nacke", "81680005", "back", 0, None, "back"),
    ("hjärtklappning", "80313002", "arrhythmia", 0, None, "arrhythmia"),
    ("skada på underben", "714825006", "knee", 0, None, "inj:S82,S81,S80"),
    ("kateterproblem", "397835001", "urinary", 0, None, "uti"),
    ("blödning", "131148009", "malaise", 0, None, "gi_bleed"),
    ("smärta i ljumske", "102555003", "abd", 0, None, "abd"),
    ("andningsuppehåll", "1023001", "breath", 1, None, "breath"),
    ("fynd som rör scrotum", "301766000", "scrotum", 0, "M", "scrotum"),
    ("komplikation under puerperiet", "200237000", "preg", 0, "F1850", "preg"),
    ("urogenitala symtom [400076009]+kvinnligt kön", "248152002", "urinary", 0, "F", "uti"),
]
N_COMPLAINT_COMBOS = 150

# list F label -> (concept group, style); style "plain", "abbrev" or "any"; group None = std-null placeholder
NONSTD_MAP = {
    "Buksmärta": ("abd", "plain"), "Bröstsmärta": ("chest", "plain"), "Andningsbesvär": ("breath", "plain"),
    "AKUTBUK": ("abd", "abbrev"), "Huvudvärk": ("headache", "plain"), "Yrsel": ("dizzy", "plain"),
    "Extremitet svullnad/värk": ("ext", "plain"), "BRSM": ("chest", "abbrev"), "Feber": ("fever", "plain"),
    "Huvudskada": ("headinj", "plain"), "Rytmrubbning": ("arrhythmia", "plain"),
    "Graviditetsrelaterade besvär": ("preg", "any"), "Ryggvärk": ("back", "plain"), "DYSPNE": ("breath", "abbrev"),
    "Diagnosinformation saknas": (None, "any"), "Urinvägsbesvär": ("urinary", "plain"), "Sårskada": ("wound", "plain"),
    "Sjukdomskänsla": ("malaise", "any"), "Höftskada": ("hip", "any"), "Vaginal blödning": ("vag", "any"),
    "Fotskada": ("foot", "plain"), "Knäskada": ("knee", "any"), "HJÄRTA": ("chest", "abbrev"),
    "Axelskada": ("shoulder", "any"), "INF": ("fever", "abbrev"), "TRHAND": ("hand", "any"),
    "Halsont": ("throat", "any"), "Flanksmärta": ("flank", "any"), "YRSEL": ("dizzy", "abbrev"),
    "BUK": ("abd", "abbrev"), "Svimning": ("syncope", "any"), "TRHUVUD": ("headinj", "abbrev"),
    "Förgiftning": ("poison", "plain"), "Handledsskada": ("wrist", "any"), "BORTFALL": (None, "abbrev"),
    "Extremitet svullnad, värk": ("ext", "plain"), "TRIAGE": (None, "any"), "Fotledsskada": ("ankle", "any"),
    "Buk": ("abd", "plain"), "Handskada": ("hand", "plain"), "Allergisk reaktion": ("allergy", "plain"),
    "Bröstsmärtor": ("chest", "plain"), "TRFOT": ("foot", "abbrev"),
    "Ospecificerad kontaktorsak": (None, "any"), "EXTSMÄRT": ("ext", "abbrev"),
}
NONSTD_TAIL_SUFFIX = ["", "_", " övr", ", ospec"]
NONSTD_TAIL_SUFFIX_P = np.array([0.55, 0.25, 0.12, 0.08])

ARRIVAL_WALK = {"Gående", "Färdtjänst", "Polis", "Inkommer med polis", "Internt", "Taxi",
                "Konsult (inneliggande SÖS)", "Larm Gående", "Hänvisad från 1177", "Kriminalvård", "Larm Polis",
                "Larm gående övrigt", "Larm gående Trauma B"}

# exit classes
HOME, ADMIT, TRANSFER, LEFT, MISSING = range(5)
EXIT_CLASS_SHARE = np.array([0.700, 0.233, 0.034, 0.020, 0.013])
EXIT_CLASS_OF = {
    "Hem": HOME, "hem": HOME, "Hemmet": HOME, "30 - Från/Till hemmet": HOME, "30 Utskriven till hem": HOME,
    "Avvikit": LEFT, "Avviker": LEFT, "Registrering saknas": MISSING,
    "Annan inrättning": TRANSFER, "Annat sjukhus": TRANSFER, "Hänvisad": TRANSFER, "Annat": TRANSFER,
    "Annan vårdenhet": TRANSFER, "33 - PV-bokad på VC/JC": TRANSFER, "Satellit": TRANSFER,
}


def exit_hospital(s):
    if s.startswith("DS "):
        return DS
    if s.startswith(("H - ", "S - ")):
        return KS
    if s == "Inlagda sös":
        return SOS
    if s.startswith(("30 - ", "39 - ", "33 - ")):
        return SU
    if s.startswith(("STSAB", "Inskriven avd", "30 Utskriven", "39 Direkt")):
        return STG
    return -1


EXIT_TAIL_ADMIT = {
    KS: [f"{p} - {w}" for p in "HS" for w in ["MAVA M82 Medicin", "KAVA K42 Kirurg", "Avd K44", "Avd R54",
                                               "Thorax T12", "Avd N26", "Infektion I73", "Avd M86 Geriatrik",
                                               "Urolog avd K74", "Hjärt avd N35", "IVA", "Avd B76"]],
    SOS: [f"Inlagda sös avd {n}" for n in [10, 21, 23, 33, 41, 44, 52, 55, 61, 63, 72, 81, 83, 93]],
    SU: [f"39 - {w}" for w in ["Till IVA", "Till annat sjukhus slutenvård", "Till obs", "Till op",
                               "Till slutenvård Mölndal", "Till slutenvård Östra", "Till geriatrik", "Till psykiatri"]],
    DS: [f"DS {w}" for w in ["Medicin 71", "Medicin 72", "Medicin 74", "Kirurgi avd 62", "Kirurgi avd 64",
                             "Ortopedi avd 82", "Ortopedi avd 83", "Hjärt avd 93", "Infektion avd 21", "Geriatrik avd 31",
                             "Gynekologi avd 15", "IVA"]],
    STG: [f"STSAB - {w}" for w in ["Med avd 6", "Med avd 7", "Med avd 9", "Kir avd 4", "Kir avd 5", "Ort avd 12",
                                    "Inf avd 21", "Ger avd 31"]] + [f"Inskriven avd {n}" for n in [30, 34, 37, 40, 41, 52, "32M", "51S"]],
    **{h: [f"Avd {n} {w}" for n, w in zip(range(10 + 4 * h, 22 + 4 * h), ["Medicin", "Kirurgi", "Ortopedi", "Hjärt",
                                                                       "Infektion", "Geriatrik", "Neurologi", "Lung",
                                                                       "Njur", "IVA", "Gyn", "Urologi"])]
       for h in range(5, 12)},
}
EXIT_TAIL_HOME = ["Hem med hemtjänst", "Särskilt boende", "Korttidsboende", "Hem/egen bostad", "Hem via apotek",
                  "Hemsjukvård", "Servicehus", "Hem med uppföljning", "Hem utan åtgärd", "Gruppboende"]
EXIT_TAIL_TRANSFER = ["Vårdcentral", "Primärvård", "Psykiatrin", "Geriatriken", "Närakut", "Jourcentral",
                      "Annan klinik", "Barnakuten", "Ögonakuten", "ÖNH-akuten", "Kvinnokliniken", "Beroendevård",
                      "Rättspsykiatri", "Polisen", "Sobringsenhet", "Tillnyktringsenhet", "Hänvisad tandläkare",
                      "Hänvisad VC", "Annat landsting", "Utlandet"]

EXIT_CU_HOSP = {"DS Medicin": DS, "DS Hjärt": DS, "DS Kirurgi": DS, "DS Ortopedi": DS, "IntmedAVA": SOS,
                "Obsenheten": SOS, "DALEN": SOS, "KirAVA": SOS, "Avdelning 4": STG, "Avdelning 3": STG,
                "Avdelning 2": STG, "Avdelning 1": STG}
EXIT_CU_TAIL = {
    KS: [f"{p}{n}" for p in ("K", "M", "R", "N") for n in range(10, 30)],
    SOS: [f"Avd {n}" for n in range(10, 40)] + ["IntkirAVA", "Obs 2", "Neuro AVA"],
    SU: [str(n) for n in list(range(10, 23)) + list(range(26, 31)) + list(range(37, 50)) + list(range(58, 66))],
    DS: ["DS Infektion", "DS Geriatrik", "DS Gynekologi", "DS Neurologi", "DS IVA", "DS Rehab", "DS Lung"],
    STG: [f"Avdelning {n}" for n in range(5, 25)],
    **{h: [f"Avd {n}" for n in range(1, 25)] for h in range(5, 12)},
}
EXIT_CU_GENERIC_TAIL = ["IVA", "Psyk", "HIA", "Op", "Röntgen", "Uppvak", "Barnakut", "Förlossning"]

# =============================================================================
# ICD-10, KVÅ, EKOD, hospital code pools (register long tails)
# =============================================================================

C_BLOCKS = ["C00-C14", "C15-C26", "C30-C39", "C40-C41", "C43-C44", "C45-C49", "C50-C50", "C51-C58",
            "C60-C63", "C64-C68", "C69-C72", "C73-C75", "C76-C80", "C81-C96", "C97-C97"]
OTHER_BLOCKS = [
    "D00-D09", "D10-D36", "D37-D48",
    "G00-G09", "G10-G14", "G20-G26", "G30-G32", "G35-G37", "G40-G47", "G50-G59", "G60-G64", "G70-G73",
    "G80-G83", "G90-G99",
    "H00-H06", "H10-H13", "H15-H22", "H25-H28", "H30-H36", "H40-H42", "H43-H45", "H46-H48", "H49-H52",
    "H53-H54", "H55-H59", "H60-H62", "H65-H75", "H80-H83", "H90-H95",
    "L00-L08", "L10-L14", "L20-L30", "L40-L45", "L50-L54", "L55-L59", "L60-L75", "L80-L99",
    "O00-O08", "O10-O16", "O20-O29", "O30-O48", "O60-O75", "O80-O84", "O85-O92", "O94-O99",
    "R00-R09", "R10-R19", "R20-R23", "R25-R29", "R30-R39", "R40-R46", "R47-R49", "R50-R69", "R70-R79",
    "R80-R82", "R83-R89", "R90-R94", "R95-R99",
]
FOUR_CHAR_STEMS = {
    "E": [3, 4, 5, 6, 7, 10, 11, 13, 14, 16, 21, 22, 23, 27, 44, 46, 53, 55, 63, 66, 78, 83, 86, 87, 88, 89],
    "D": [50, 51, 52, 53, 59, 61, 62, 63, 64, 65, 68, 69, 70, 72, 75, 80, 84, 86, 89],
    "I": [10, 11, 12, 13, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 33, 34, 35, 36, 38, 40, 42, 44, 45, 46, 47, 48,
          49, 50, 51, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 77, 78, 80, 81, 82, 83, 84, 85, 86,
          87, 88, 89, 95, 97, 99],
}


def block_of_c(code3: str) -> str:
    n = int(code3[1:3])
    for b in C_BLOCKS:
        if int(b[1:3]) <= n <= int(b[5:7]):
            return b
    return "C76-C80"


def build_icd_pool():
    pool = list(C_BLOCKS) + list(OTHER_BLOCKS)
    for ch, (a, b) in {"A": (0, 99), "B": (0, 99), "F": (0, 99), "J": (0, 99), "K": (0, 93), "M": (0, 99),
                       "N": (0, 99), "S": (0, 99), "T": (0, 98), "Z": (0, 99), "P": (0, 19), "Q": (20, 39)}.items():
        pool += [f"{ch}{i:02d}" for i in range(a, b + 1)]
    for ch, stems in FOUR_CHAR_STEMS.items():
        pool += [f"{ch}{s:02d}{d}" for s in stems for d in range(10)]
    pool += ["E871B", "E870A", "I489A", "I489B", "I489C", "I109A", "I251A", "E119X", "I639B", "I219A",
             "U821", "U822", "U070", "U071", "U099", "U129", "U812", "U898"]
    return list(dict.fromkeys(pool))


def register_format(code: str) -> str:
    """Death-register ICD-10 code -> patient-register granularity."""
    ch = code[0]
    if ch == "C":
        return block_of_c(code[:3])
    if ch == "D" and int(code[1:3]) < 50:
        return "D00-D09" if int(code[1:3]) < 10 else ("D10-D36" if int(code[1:3]) < 37 else "D37-D48")
    if ch in "GHLOR":
        blocks = [b for b in OTHER_BLOCKS if b[0] == ch]
        n = int(code[1:3])
        for b in blocks:
            if int(b[1:3]) <= n <= int(b[5:7]):
                return b
        return blocks[-1]
    if ch in "EI" or (ch == "D"):
        return (code[:4] + "9")[:4] if len(code) < 4 else code[:4]
    return code[:3]


def build_kva_pool(rng):
    two = ["AA", "AB", "AC", "AD", "AF", "AG", "AH", "AJ", "AK", "AL", "AM", "AN", "AP", "AQ", "AU", "AV", "AW",
           "DA", "DB", "DD", "DF", "DG", "DH", "DJ", "DK", "DL", "DN", "DP", "DR", "DT", "DV", "GA", "GB", "NB",
           "NC", "ND", "NF", "NG", "NH", "PE", "PH", "QA", "QB", "QD", "TA", "TB", "TJ", "TK", "TL", "TN", "TP",
           "XA", "XB", "XC", "XD", "XF", "XJ", "XK", "XN", "XS", "XT", "XU", "XV", "XX", "ZV", "ZX", "ZZ"]
    pool = [f"{p}{n:03d}" for p in two for n in rng.choice(1000, 40, replace=False)]
    first = list("ABCDEFGHJKLMNPQTU")
    second = list("ABCDEFGHJKLMN")
    third = list("ABCDEFGHJKLMNPQRSTUW")
    combos = set()
    while len(combos) < 3500:
        combos.add(f"{rng.choice(first)}{rng.choice(second)}{rng.choice(third)}{rng.integers(0, 100):02d}")
    return pool + sorted(combos)


def build_ekod_pool():
    pool = []
    for ch, (a, b) in (("W", (0, 99)), ("X", (0, 59))):
        for i in range(a, b + 1):
            for place in "0129":
                for act in "0129":
                    pool.append(f"{ch}{i:02d}{place}{act}")
    pool += [f"X{i:02d}{p}9" for i in range(60, 85) for p in "09"]
    pool += [f"Y{i:02d}{p}9" for i in range(0, 35) for p in "09"]
    pool += [f"V{i:02d}{d}{p}" for i in range(1, 90) for d in "49" for p in "09"]
    pool += [f"Y{i}{d}" for i in range(40, 90) for d in range(10)]
    return pool


def build_sjukhus_pool(rng, exclude):
    prefixes = ["10", "11", "12", "13", "21", "22", "23", "24", "25", "26", "27", "28", "30", "31", "41", "42",
                "50", "51", "52", "53", "54", "55", "56", "57", "61", "62", "63", "64", "65"]
    pool = set()
    while len(pool) < 1077 - 38:
        c = f"{rng.choice(prefixes)}{rng.integers(0, 1000):03d}"
        if c not in exclude:
            pool.add(c)
    return sorted(pool)


def icd_sex_rule(code: str):
    """Allowed in (male, female 15-50, female other)."""
    if code.startswith("O") or code in [f"Z3{i}" for i in range(10)]:
        return (False, True, False)
    if code in ("C51-C58",) or (code[0] == "N" and len(code) == 3 and 70 <= int(code[1:3]) <= 98):
        return (False, True, True)
    if code == "C60-C63" or (code[0] == "N" and len(code) == 3 and 40 <= int(code[1:3]) <= 51):
        return (True, False, False)
    return (True, True, True)


DX_MAP = {
    "abd": [("R10-R19", .70), ("K35", .06), ("K80", .08), ("K56", .05), ("K57", .06), ("K92", .05)],
    "chest": [("R00-R09", .70), ("I209", .10), ("I219", .10), ("I489", .10)],
    "breath": [("R00-R09", .40), ("J44", .20), ("J18", .20), ("I509", .20)],
    "fever": [("R50-R69", .50), ("J18", .25), ("N39", .25)],
    "uti": [("N39", .70), ("N30", .20), ("R30-R39", .10)],
    "neuro": [("R40-R46", .35), ("R50-R69", .35), ("G40-G47", .30)],
    "arrhythmia": [("I489", .70), ("R00-R09", .30)],
    "poison": [("T42", .30), ("T51", .30), ("T40", .20), ("F10", .20)],
    "allergy": [("T78", .60), ("L50-L54", .40)],
    "preg": [("O20-O29", .60), ("O00-O08", .40)],
    "vag": [("O20-O29", .50), ("N93", .50)],
    "stroke": [("I639", .60), ("G40-G47", .20), ("I619", .20)],
    "seizure": [("G40-G47", 1.0)],
    "flank": [("N20", .60), ("R10-R19", .40)],
    "throat": [("J02", .50), ("J03", .30), ("J36", .20)],
    "diarr": [("A09", .60), ("K52", .40)],
    "epistaxis": [("R00-R09", 1.0)],
    "back": [("M54", .80), ("M51", .20)],
    "ext": [("I802", .30), ("M79", .50), ("R20-R23", .20)],
    "malaise": [("R50-R69", .70), ("R40-R46", .30)],
    "psych": [("F41", .40), ("F32", .30), ("F43", .30)],
    "alcohol": [("F10", 1.0)],
    "gi_bleed": [("K92", .70), ("K25", .30)],
    "diab": [("E119", .50), ("E162", .50)],
    "htn": [("I109", 1.0)],
    "fb": [("T17", .40), ("T18", .40), ("T15", .20)],
    "dental": [("K04", .50), ("K08", .50)],
    "eye": [("H10-H13", .50), ("H53-H54", .50)],
    "ear": [("H60-H62", .50), ("H65-H75", .50)],
    "skin": [("L20-L30", .50), ("L50-L54", .50)],
    "skin_inf": [("L00-L08", 1.0)],
    "scrotum": [("N45", .50), ("N50", .50)],
}

# Death register pools (full ICD-10 codes, mixed precision)
DEATH_STEMS = (
    "A04 A08 A09 A40 A41 A49 B18 B20 C01 C02 C10 C13 C15 C16 C17 C18 C19 C20 C21 C22 C23 C24 C25 C32 C34 C38 C43 "
    "C44 C45 C48 C49 C50 C51 C53 C54 C55 C56 C57 C61 C62 C64 C65 C66 C67 C68 C69 C70 C71 C73 C74 C76 C77 C78 C79 "
    "C80 C81 C82 C83 C84 C85 C88 C90 C91 C92 C93 C94 C95 D32 D33 D43 D46 D47 D50 D61 D64 D65 D68 D69 E03 E05 E10 "
    "E11 E13 E14 E43 E46 E86 E87 F01 F03 F05 F10 F20 F32 G12 G20 G23 G30 G31 G35 G40 G41 G45 G61 G70 G93 G95 I05 "
    "I08 I10 I11 I12 I13 I20 I21 I24 I25 I26 I27 I33 I34 I35 I38 I42 I44 I45 I46 I47 I48 I49 I50 I51 I60 I61 I62 "
    "I63 I64 I67 I69 I70 I71 I72 I74 I77 I80 I95 J10 J12 J13 J15 J18 J40 J43 J44 J45 J47 J69 J80 J81 J84 J90 J93 "
    "J96 J98 K22 K25 K26 K29 K31 K40 K55 K56 K57 K63 K65 K70 K72 K74 K76 K80 K81 K83 K85 K92 L89 M06 M15 M16 M17 "
    "M31 M32 M48 M62 M80 M81 N17 N18 N19 N28 N39 N40 Q20 Q21 Q24 R00 R02 R06 R09 R13 R26 R40 R41 R50 R52 R53 R54 "
    "R57 R63 R64 R68 R69 R95 R96 R98 R99"
).split()
DEATH_EXT_POOL = (
    [f"V{i:02d}{d}" for i in (3, 4, 9, 23, 28, 43, 47, 48, 49, 87, 89) for d in "049"]
    + ["W00", "W01", "W05", "W06", "W08", "W10", "W13", "W17", "W18", "W19", "W190", "W199", "W01X", "W69", "W74",
       "W78", "W79", "W80", "W84"]
    + [f"X{i}{d}" for i in range(40, 50) for d in "09"] + ["X590", "X599", "X31", "X00", "X47"]
    + [f"X{i}{d}" for i in range(60, 85) for d in "09"] + [f"Y{i:02d}{d}" for i in range(10, 35) for d in "09"]
    + ["Y832", "Y838", "Y839", "Y848", "Y86", "Y88", "Y95"]
)


def build_death_pool():
    pool = []
    for s in DEATH_STEMS:
        pool.append(s)
        for d in "0189":
            pool.append(s + d)
    pool += ["I5090", "I5099", "I5010", "I5000", "I2190", "I6390", "J1890", "C3490"]
    return list(dict.fromkeys(pool))


def is_external(code: str) -> bool:
    return code[0] in "VWXY"


def death_sex_rule(code: str):
    """Allowed in (male, female)."""
    c3 = code[:3]
    if c3 in ("C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58") or code.startswith("O") or c3 == "N7":
        return (False, True)
    if c3 in ("C60", "C61", "C62", "C63") or c3 == "N40":
        return (True, False)
    return (True, True)


KAP19_POOL = ([f"S{i:02d}" for i in (0, 1, 2, 6, 7, 12, 13, 22, 27, 32, 36, 42, 72, 82)]
              + [f"S{i:02d}{d}" for i in (6, 12, 22, 27, 32, 72) for d in "0189"]
              + [f"S{i:02d}{d}0" for i in (6, 72) for d in "0123"]
              + [f"T{i:02d}{d}" for i in (17, 30, 39, 40, 42, 43, 44, 45, 50, 51, 59, 65, 68, 71, 75, 79, 81, 88)
                 for d in "0129"]
              + ["T07", "T14", "T149", "T71", "T58", "T68", "T17", "T31", "T509"])

# =============================================================================
# Lab terms (section 7, list U plus assumed other terms)
# =============================================================================

# name, term_id, unit, ref women, ref men, value distribution, kind
MAIN_TERMS = [
    ("Blodtryck systoliskt - övre", 3719, None, None, None, ("norm", 140, 25, 60, 260, 0), "vital"),
    ("Blodtryck diastoliskt - nedre", 3720, None, None, None, ("norm", 80, 14, 30, 150, 0), "vital"),
    ("Saturation", 2025, None, None, None, ("sat",), "vital"),
    ("Andningsfrekvens", 402, None, None, None, ("norm", 18, 4, 6, 50, 0), "vital"),
    ("Kroppstemperatur", 1995, None, None, None, ("norm", 37.0, 0.7, 34.0, 41.5, 1), "vital"),
    ("Pulsfrekvens", 1978, None, None, None, ("norm", 85, 18, 30, 200, 0), "vital"),
    ("Medvetandegrad enl Retts", 1990, None, None, None, ("codes", [1, 2, 3, 4], [.92, .05, .02, .01]), "vital"),
    ("B-Leukocyter", 2005, "x10(9)/L", "3,5-8,8", "3,5-8,8", ("lnorm", 9.0, 18, 0.5, 150, 1), "lab"),
    ("B-Hemoglobin", 2006, "g/L", "117-153", "134-170", ("normsex", (128, 17), (138, 19), 40, 220, 0), "lab"),
    ("B-Trombocyter", 2007, "x10(9)/L", "165-387", "145-348", ("lnorm", 245, 450, 5, 1500, 0), "lab"),
    ("B-EVF", 2008, None, "0,35-0,46", "0,39-0,50", ("normsex", (0.38, 0.05), (0.41, 0.05), 0.12, 0.70, 2), "lab"),
    ("P-Kreatinin", 2010, "mikromol/L", "<90", "<100", ("lnorm", 80, 200, 20, 1500, 0), "lab"),
    ("P-CRP", 2012, "mg/L", "<3", "<3", ("crp",), "lab"),
    ("P-Kalium", 2100, "mmol/L", "3,5-4,6", "3,5-4,6", ("norm", 4.0, 0.45, 2.0, 8.0, 1), "lab"),
    ("P-Natrium", 2101, "mmol/L", "137-145", "137-145", ("norm", 139, 4, 110, 170, 0), "lab"),
    ("P-Glukos", 2102, "mmol/L", "4,0-6,0", "4,0-6,0", ("lnorm", 6.5, 13, 1.5, 50, 1), "lab"),
]
VITAL_ROUND_P = {"Blodtryck systoliskt - övre": 1.0, "Blodtryck diastoliskt - nedre": 0.937, "Saturation": 0.922,
                 "Andningsfrekvens": 0.850, "Kroppstemperatur": 0.789, "Pulsfrekvens": 0.526}
LAB_PANEL_P = {"B-Leukocyter": 1.0, "B-Hemoglobin": 0.9987, "B-Trombocyter": 0.9949, "B-EVF": 0.9894,
               "P-Kreatinin": 0.9721, "P-CRP": 0.9709, "P-Kalium": 0.8644, "P-Natrium": 0.8612, "P-Glukos": 0.7196}
P_LAB_VISIT = 0.908
OTHER_TERMS_SHARE = 0.189
NEWS_SHARE_OF_OTHER = 0.16

# name, unit, ref women, ref men, relative weight, distribution, kind  (ASSUMPTION: names and weights)
OTHER_TERMS = [
    ("B-Laktat", "mmol/L", "0,5-2,3", "0,5-2,3", 100910, ("lnorm", 1.6, 4.5, 0.3, 25, 1), "lab"),
    ("B-Natrium", "mmol/L", "137-145", "137-145", 101858, ("norm", 139, 4, 110, 170, 0), "lab"),
    ("B-Kalium", "mmol/L", "3,6-4,6", "3,6-4,6", 101811, ("norm", 4.0, 0.45, 2, 8, 1), "lab"),
    ("B-Glukos", "mmol/L", "4,0-6,0", "4,0-6,0", 154029, ("lnorm", 6.8, 14, 1.5, 50, 1), "lab"),
    ("P-Standardbikarbonat", "mmol/L", "22-27", "22-27", 100322, ("norm", 24, 3, 5, 45, 1), "lab"),
    ("B-Base excess", "mmol/L", "-3-3", "-3-3", 57012, ("norm", 0, 3.5, -30, 20, 1), "lab"),
    ("P-Klorid", "mmol/L", "98-107", "98-107", 48516, ("norm", 103, 4, 80, 130, 0), "lab"),
    ("P-Urea", "mmol/L", "<8", "<8", 88836, ("lnorm", 6.5, 18, 1, 60, 1), "lab"),
    ("P-Anjongap", "mmol/L", "8-16", "8-16", 85137, ("norm", 12, 3, 0, 40, 0), "lab"),
    ("P-Joniserat kalcium", "mmol/L", "1,15-1,33", "1,15-1,33", 20000, ("norm", 1.22, 0.07, 0.7, 1.8, 2), "lab"),
    ("P-Kalcium", "mmol/L", "2,15-2,50", "2,15-2,50", 15000, ("norm", 2.3, 0.15, 1.4, 3.5, 2), "lab"),
    ("B-Neutrofila", "x10(9)/L", "1,6-5,9", "1,6-5,9", 17569, ("lnorm", 6.5, 15, 0.1, 60, 1), "lab"),
    ("B-Hb (blodgas)", "g/L", "120-150", "130-160", 100077, ("normsex", (128, 17), (138, 19), 40, 220, 0), "lab"),
    ("P-Albumin", "g/L", "34-45", "34-45", 33143, ("norm", 36, 5, 10, 55, 0), "lab"),
    ("P-Bilirubin", "mikromol/L", "<26", "<26", 107168, ("lnorm", 10, 40, 2, 500, 0), "lab"),
    ("P-Urat", "mikromol/L", "155-350", "230-480", 10195, ("norm", 330, 90, 80, 900, 0), "lab"),
    ("P-D-dimer", "mg/L", "<0,25", "<0,25", 22026, ("lnorm", 0.6, 5, 0.1, 35, 2), "lab"),
    ("B-CRP (patientnära)", "mg/L", None, None, 89876, ("lnorm", 15, 180, 5, 250, 0), "lab"),
    ("P-ALAT", "mikrokat/L", "<0,76", "<1,1", 100000, ("lnorm", 0.45, 2.0, 0.05, 40, 2), "lab"),
    ("P-ASAT", "mikrokat/L", "<0,61", "<0,61", 54047, ("lnorm", 0.5, 2.5, 0.1, 50, 2), "lab"),
    ("P-Pankreasamylas", "mikrokat/L", "<1,5", "<1,5", 49089, ("lnorm", 0.4, 1.6, 0.05, 30, 2), "lab"),
    ("P-ALP", "mikrokat/L", "<2,0", "<2,0", 49084, ("lnorm", 1.3, 3.5, 0.2, 30, 1), "lab"),
    ("P-CK", "mikrokat/L", "<3,5", "<3,5", 26553, ("lnorm", 2.0, 15, 0.1, 500, 1), "lab"),
    ("P-GT", "mikrokat/L", "<0,75", "<1,2", 40000, ("lnorm", 0.5, 3.0, 0.05, 40, 2), "lab"),
    ("P-Prokalcitonin", "mikrog/L", "<0,5", "<0,5", 30000, ("lnorm", 0.12, 2.5, 0.02, 100, 2), "lab"),
    ("B-Basofila", "x10(9)/L", "<0,1", "<0,1", 8000, ("lnorm", 0.03, 0.1, 0.0, 1, 2), "lab"),
    ("B-Monocyter", "x10(9)/L", "0,2-0,8", "0,2-0,8", 100000, ("norm", 0.6, 0.15, 0.05, 3, 2), "lab"),
    ("B-Eosinofila", "x10(9)/L", "<0,5", "<0,5", 17000, ("lnorm", 0.1, 0.5, 0.0, 5, 2), "lab"),
    ("P-Troponin T", "nanog/L", "<15", "<15", 140542, ("trop",), "lab"),
    ("aB-pCO2", "kPa", "4,6-6,0", "4,6-6,0", 40914, ("norm", 5.3, 1.0, 2, 15, 1), "lab"),
    ("vB-pCO2", "kPa", "5,3-6,6", "5,3-6,6", 61650, ("norm", 6.0, 1.1, 2, 15, 1), "lab"),
    ("vB-pH", None, "7,32-7,43", "7,32-7,43", 61320, ("norm", 7.38, 0.06, 6.8, 7.7, 2), "lab"),
    ("aB-pH", None, "7,35-7,45", "7,35-7,45", 40126, ("norm", 7.40, 0.06, 6.8, 7.7, 2), "lab"),
    ("aB-Syrgasmättnad", "%", "95-99", "95-99", 98211, ("sat",), "lab"),
    ("P-APTT", "s", "28-40", "28-40", 44857, ("lnorm", 31, 48, 15, 180, 0), "lab"),
    ("Syrgasflöde", "L/min", None, None, 40002,
     ("codes", [1, 2, 3, 4, 5, 6, 8, 10, 15], [.25, .25, .15, .1, .08, .06, .05, .04, .02]), "obs"),
    ("P-NT-proBNP", "ng/L", "<222", "<194", 27661, ("lnorm", 600, 15000, 10, 35000, 0), "lab"),
    ("B-SR", "mm", "<20", "<15", 6699, ("lnorm", 18, 70, 1, 140, 0), "lab"),
    ("P-Ferritin", "mikrog/L", "10-150", "30-400", 5427, ("lnorm", 150, 900, 5, 5000, 0), "lab"),
    ("S-Fritt T4", "pmol/L", "12-22", "12-22", 2027, ("norm", 16, 3, 5, 60, 1), "lab"),
    ("U-Erytrocyter", "antal/synf", None, None, 384, ("codes", list(range(21)), [1 / 21] * 21), "lab"),
    ("PK(INR)", None, "<1,2", "<1,2", 90000, ("lnorm", 1.1, 2.8, 0.8, 10, 1), "lab"),
    ("Smärtskattning NRS", None, None, None, 380000,
     ("codes", list(range(11)), [.08, .07, .10, .13, .12, .13, .11, .10, .08, .04, .04]), "obs"),
    ("Glasgow Coma Scale", None, None, None, 100000,
     ("codes", list(range(3, 16)), [.01] * 8 + [.01, .02, .03, .07, .82]), "obs"),
    ("Vikt", None, None, None, 120000, ("norm", 76, 17, 35, 200, 0), "obs"),
    ("Längd", None, None, None, 80000, ("norm", 171, 10, 140, 205, 0), "obs"),
]
NEWS_TERM = ("NEWS totalpoäng", None, None, None, 0,
             ("codes", list(range(16)), [.34, .16, .12, .09, .07, .05, .04, .03, .03, .02, .01, .01, .01, .005, .003, .002]),
             "obs")
OTHER_FLAGS = ["hemolys", "insuf_amount", "clot", "error", "comment", "not_performed", "lipemic", "icteric", "wrong_tube"]
OTHER_FLAG_W = np.array([13280, 9795, 7406, 2866, 2357, 1016, 9, 8, 8], float)
VALUE_STR = ["0-3", "4-10", ">10", "Neg"]
VALUE_STR_P = np.array([176, 60, 40, 18], float)

LAB_DATE_KNOTS = [(0.0, "2005-01-01"), (0.05, "2012-02-22"), (0.25, "2013-11-25"), (0.50, "2015-08-18"),
                  (0.75, "2016-09-07"), (0.95, "2017-06-13"), (1.0, "2017-08-31")]

# =============================================================================
# Section 3: persons and their deaths
# =============================================================================


def make_persons(rng, n):
    ids = np.sort(rng.choice(ID_MAX, size=n, replace=False) + 1).astype(np.float64)
    male = rng.random(n) < P_MALE
    birth = np.floor(qsample(rng, n, knots(BIRTH_KNOTS))).astype(np.int64)
    by, _, _ = ymd(birth)
    jan1 = rng.random(n) < P_BIRTH_JAN1
    birth[jan1] = from_ymd(by[jan1], 1, 1)
    z = rng.standard_normal(n)
    foreign = rng.random(n) < P_FOREIGN
    home = rng.choice(len(HOSPITALS), size=n, p=HOSP_P)
    adult = add_years(birth, 18)
    cap = add_years(birth, 110) - 1  # nobody is older than 109 at an event
    ed_lo = np.maximum(ED_START, adult)

    age2010 = (D("2010-07-01") - birth) / 365.25
    must_die = cap < DEATH_END
    lin = DEATH_AGE_COEF * (age2010 - 60) + DEATH_Z_COEF * z

    def p_of(a):
        p = 1 / (1 + np.exp(-(a + lin)))
        p[must_die] = 1.0
        return p

    a = bisect(lambda a: p_of(a).mean() - P_DEATH, -40, 10)
    die = rng.random(n) < p_of(a)
    death = np.full(n, NO_DEATH, np.int64)
    di = np.flatnonzero(die)
    death_kn = knots([(0.0, "2003-03-19"), (0.05, "2006-01-05"), (0.25, "2009-09-14"), (0.5, "2012-11-30"),
                      (0.75, "2015-07-10"), (0.95, "2017-06-25"), (1.0, "2017-12-31")])
    death[di] = qsample_days(rng, len(di), death_kn, ed_lo[di], np.minimum(DEATH_END, cap[di]))

    # inexact death dates: mostly year-only in the last year (coding lag)
    n_inexact = int(round(DEATH_INEXACT * len(di)))
    n_year = int(round(DEATH_YEAR_ONLY * n_inexact))
    last_year = di[death[di] >= D("2017-01-01")]
    k2017 = min(len(last_year), int(round(0.8 * n_year)))
    year_only = np.concatenate([rng.choice(last_year, k2017, replace=False),
                                rng.choice(np.setdiff1d(di, last_year), n_year - k2017, replace=False)])
    rest = np.setdiff1d(di, year_only)
    day_only = rng.choice(rest, n_inexact - n_year, replace=False)
    inexact = np.zeros(n, np.int8)  # 0 exact, 1 day unknown, 2 month and day unknown
    inexact[day_only] = 1
    inexact[year_only] = 2

    ed_hi = np.minimum(np.minimum(ED_END, death), cap)
    return dict(n=n, id=ids, male=male, birth=birth, z=z, foreign=foreign, home=home, cap=cap,
                ed_lo=ed_lo, ed_hi=ed_hi, die=die, death=death, inexact=inexact)


# =============================================================================
# Complaint catalogue
# =============================================================================


def build_complaints(rng):
    vals, counts, _ = parse_list(LIST_A)
    terms, info = [], []
    for v in vals:
        name = v.split(" [")[0]
        terms.append(v)
        info.append(COMPLAINT_INFO[name])
    single_tail = []
    for name, sct, group, urgent, sex, dx in COMPLAINT_TAIL:
        single_tail.append((f"{name} [{sct}]", (group, urgent, sex, dx)))
    # combinations of top terms in the same format, joined with '+'
    singles = [i for i, v in enumerate(vals) if "+" not in v][:25]
    combos = []
    seen = set(vals)
    while len(combos) < N_COMPLAINT_COMBOS:
        i, j = rng.choice(singles, 2, replace=False)
        si, sj = info[i][2], info[j][2]
        if si and sj and si != sj:
            continue
        s = f"{vals[i]}+{vals[j]}"
        if s in seen:
            continue
        seen.add(s)
        g, u, sex, dx = info[i]
        combos.append((s, (g, max(u, info[j][1]), si or sj, dx)))
    tail = single_tail + combos
    tail_order = rng.permutation(len(tail))
    tail = [tail[i] for i in tail_order]
    tw = capped_zipf(len(tail), (1 - 0.1658) - counts.sum() / REAL_ROWS["ed"], counts.min() / REAL_ROWS["ed"] * 0.9,
                     s=0.9)
    terms += [t for t, _ in tail]
    info += [i for _, i in tail]
    weights = np.concatenate([counts / REAL_ROWS["ed"], tw])
    groups = sorted({i[0] for i in info})
    gidx = vocab_index(groups)
    sex_allowed = np.array([[i[2] in (None, "M"), i[2] in (None, "F", "F1850"), i[2] in (None, "F")] for i in info]).T
    return dict(terms=terms, weights=weights, group=np.array([gidx[i[0]] for i in info]), groups=groups,
                urgent=np.array([i[1] for i in info]), dx=[i[3] for i in info], allowed=sex_allowed)


# =============================================================================
# ED visits (shared by ED, patient register and lab)
# =============================================================================


def make_visits(rng, P, C):
    n = P["n"]
    lo, hi = P["ed_lo"], P["ed_hi"]
    male = P["male"]
    ed_kn = knots(ED_DATE_KNOTS)
    expo = cdf(hi + 1.0, ed_kn) - cdf(lo, ed_kn)
    amid = ((lo + hi) / 2 - P["birth"]) / 365.25
    base = expo * np.exp(ED_Z_SLOPE * P["z"] + ED_AGE_SLOPE * (amid - 50))
    k = ED_NB_K
    target = ROWS_PER_PERSON["ed"] / float(np.dot([1, 2, 3], ROWS_PER_VISIT_P))

    def mean_zt(mu):
        p0 = (k / (k + mu)) ** k
        return np.where(mu < 1e-9, 1.0, mu / np.maximum(1 - p0, 1e-300))

    f = 1.0
    for _ in range(25):
        w = base * np.where(male, f, 1.0)
        c = bisect(lambda c: mean_zt(c * w).mean() - target, 1e-4, 1e4)
        e = mean_zt(c * w)
        share = e[male].sum() / e.sum()
        f *= (ED_MALE_ROW_SHARE / (1 - ED_MALE_ROW_SHARE)) / (share / (1 - share))
    cnt = draw_zt_nb(rng, c * w, k)

    vp = np.repeat(np.arange(n), cnt)
    vdate = qsample_days(rng, len(vp), ed_kn, lo[vp], hi[vp])
    first_visit = np.concatenate([[0], np.cumsum(cnt)[:-1]])

    # deaths linked to visits and register stays
    exact_dead = P["die"] & (P["inexact"] == 0)
    death = P["death"]
    n_obduk = int(round(OBDUK_ROW_SHARE * ROWS_PER_PERSON["ed"] * n))
    elig = np.flatnonzero(exact_dead & (death <= ED_END))
    obduk = np.zeros(n, bool)
    obduk[rng.choice(elig, min(n_obduk, len(elig)), replace=False)] = True
    near = (exact_dead & (death <= ED_END) & (rng.random(n) < P_NEAR_DEATH_VISIT)) | obduk
    n_inhosp = int(round(UTSATT4_SHARE * INPAT_SHARE * ROWS_PER_PERSON["patreg"] * n))
    elig = np.flatnonzero(exact_dead & ~obduk & (death <= REG_END))
    inhosp = np.zeros(n, bool)
    inhosp[rng.choice(elig, min(n_inhosp, len(elig)), replace=False)] = True

    near_p = np.flatnonzero(near)
    off = 1 + np.floor(rng.random(len(near_p)) * 30).astype(np.int64)
    off[inhosp[near_p]] = np.floor(rng.random(inhosp[near_p].sum()) * 15).astype(np.int64)
    off[obduk[near_p]] = 0
    nv = first_visit[near_p]
    vdate[nv] = np.maximum(death[near_p] - off, lo[near_p])
    v_near = np.zeros(len(vp), bool)
    v_near[nv] = True

    # deliberate error: an ED visit 1-60 days after death
    elig = np.flatnonzero(exact_dead & ~obduk & (death < ED_END))
    n_err = max(1, int(round(ERR_ED_AFTER_DEATH * P["die"].sum())))
    err_p = rng.choice(elig, min(n_err, len(elig)), replace=False)
    err_date = np.minimum(death[err_p] + 1 + np.floor(rng.random(len(err_p)) * 60).astype(np.int64), ED_END)

    vp = np.concatenate([vp, err_p])
    vdate = np.concatenate([vdate, err_date])
    v_near = np.concatenate([v_near, np.zeros(len(err_p), bool)])
    v_err = np.concatenate([np.zeros(len(vp) - len(err_p), bool), np.ones(len(err_p), bool)])
    order = np.lexsort((vdate, vp))
    vp, vdate, v_near, v_err = vp[order], vdate[order], v_near[order], v_err[order]
    nvis = len(vp)

    v_obduk = v_near & obduk[vp] & (vdate == death[vp])
    v_inhosp_near = v_near & inhosp[vp]

    hosp = np.where(rng.random(nvis) < HOME_HOSPITAL_SHARE, P["home"][vp], rng.choice(12, nvis, p=HOSP_P))
    age = age_years(P["birth"][vp], vdate)
    female = ~P["male"][vp]
    sex_group = np.where(~female, 0, np.where((age >= 18) & (age <= 50), 1, 2))
    concept = sample_grouped(rng, sex_group, C["weights"], C["allowed"])
    urgent = C["urgent"][concept]
    z = P["z"][vp]

    # internal exit class: admission rises with age, frailty and urgency
    score = 0.035 * (age - 54) + 0.5 * z + 0.9 * urgent + rng.standard_normal(nvis)
    cls = np.full(nvis, HOME)
    n_adm = int(round(EXIT_CLASS_SHARE[ADMIT] * nvis))
    cls[np.argsort(-score)[:n_adm]] = ADMIT
    free = np.flatnonzero(cls == HOME)
    left_score = -0.04 * age[free] + rng.standard_normal(len(free))
    n_left = int(round(EXIT_CLASS_SHARE[LEFT] * nvis))
    cls[free[np.argsort(left_score)[:n_left]]] = LEFT
    for c in (TRANSFER, MISSING):
        free = np.flatnonzero(cls == HOME)
        cls[rng.choice(free, int(round(EXIT_CLASS_SHARE[c] * nvis)), replace=False)] = c
    cls[v_inhosp_near] = ADMIT
    cls[v_obduk] = TRANSFER

    return dict(n=nvis, person=vp, date=vdate, near=v_near, err=v_err, obduk=v_obduk, inhosp_near=v_inhosp_near,
                hosp=hosp, age=age, concept=concept, urgent=urgent, cls=cls, z=z, sex_group=sex_group,
                p_obduk=obduk, p_inhosp=inhosp)


# =============================================================================
# Section 4: swed_ed_core
# =============================================================================

ED_SCHEMA = pa.schema([
    ("ed_hospital", pa.string()), ("lopnr_swed", pa.float64()),
    ("ed_arrival_datetime", pa.timestamp("us", tz="UTC")), ("ed_inexact_time", pa.bool_()),
    ("ed_age", pa.float64()), ("is_male", pa.int32()), ("ed_presented_complaint_std", pa.string()),
    ("ed_arrival_date", pa.date32()), ("ed_arrival_time", pa.string()),
    ("ed_time_to_doctor_minutes", pa.float64()), ("ed_time_total_minutes", pa.float64()),
    ("ed_priority_in", pa.float64()), ("ed_arrival_type", pa.string()), ("care_unit", pa.string()),
    ("ed_priority_out", pa.float64()), ("ed_exit_to", pa.string()), ("ed_exit_to_care_unit", pa.string()),
    ("ed_presented_complaint_nonstd", pa.string()),
])

HOUR_WEIGHTS = np.array([2.6, 2.0, 1.6, 1.3, 1.2, 1.3, 1.9, 3.0, 4.8, 6.4, 7.5, 7.5, 7.5, 7.5, 6.8, 6.4, 6.0, 5.6,
                         5.2, 4.8, 4.3, 3.8, 3.3, 2.8])


def str_col(idx, vocab):
    idx = np.asarray(idx, np.int64)
    mask = idx < 0
    ind = pa.array(np.where(mask, 0, idx).astype(np.int32), mask=mask)
    return pa.DictionaryArray.from_arrays(ind, pa.array(list(vocab), type=pa.string())).dictionary_decode()


def obj_col(values):
    return pa.array(values, type=pa.string())


def date_col(days, mask=None):
    return pa.array(np.asarray(days, np.int64).astype("datetime64[D]"), type=pa.date32(), mask=mask)


def ts_col(us):
    return pa.array(np.asarray(us, np.int64).astype("datetime64[us]"), type=pa.timestamp("us", tz="UTC"))


def f64_col(x):
    x = np.asarray(x, float)
    return pa.array(x, type=pa.float64(), mask=np.isnan(x))


def i32_col(x, mask=None):
    return pa.array(np.asarray(x).astype(np.int32), type=pa.int32(), mask=mask)


def time_of_day(rng, n):
    w = np.repeat(HOUR_WEIGHTS / 60, 60)
    return rng.choice(1440, size=n, p=w / w.sum())


def care_unit_model():
    vals, counts, extra = parse_list(LIST_B)
    hosp_of = [HOSPITALS.index(e[0]) for e in extra]
    total = HOSP_P * REAL_ROWS["ed"]
    vocab = list(vals)
    per_h = {}
    outside = {}
    for h in range(12):
        listed = [i for i in range(len(vals)) if hosp_of[i] == h]
        outside[h] = max(total[h] - counts[listed].sum(), 0)
        per_h[h] = listed
    null_total = 0.1147 * REAL_ROWS["ed"]
    fixed = sum(CARE_UNIT_OUTSIDE_NULL[h] * outside[h] for h in CARE_UNIT_OUTSIDE_NULL)
    rest = sum(outside[h] for h in range(12) if h not in CARE_UNIT_OUTSIDE_NULL)
    other_null = (null_total - fixed) / rest
    model = {}
    for h in range(12):
        nf = CARE_UNIT_OUTSIDE_NULL.get(h, other_null)
        rare = RARE_CARE_UNITS.get(h, [])
        ids = list(per_h[h])
        w = list(counts[per_h[h]])
        rare_total = outside[h] * (1 - nf)
        rw = capped_zipf(len(rare), rare_total, counts.min() * 0.9, s=0.8)
        for r, x in zip(rare, rw):
            vocab.append(r)
            ids.append(len(vocab) - 1)
            w.append(x)
        ids.append(-1)
        w.append(outside[h] * nf)
        model[h] = (np.array(ids), np.array(w, float))
    return vocab, model


def make_ed(rng, P, V, C):
    nvis = V["n"]
    vp = V["person"]
    # rows per visit
    nrows = 1 + choice_p(rng, nvis, ROWS_PER_VISIT_P)
    nrows[V["obduk"] | V["err"]] = 1
    inexact_v = rng.random(nvis) < 0.0513
    minute_v = np.where(inexact_v, 0, time_of_day(rng, nvis))
    V["inexact"] = inexact_v
    V["minute"] = minute_v

    r_vis = np.repeat(np.arange(nvis), nrows)
    n = len(r_vis)
    first = np.concatenate([[True], r_vis[1:] != r_vis[:-1]])
    step = np.where(first, 0, 5 + np.floor(rng.random(n) * 236).astype(np.int64))
    csum = np.cumsum(step)
    start = np.maximum.accumulate(np.where(first, csum, 0))
    minute = np.minimum(minute_v[r_vis] + (csum - start), 1439)
    inexact = inexact_v[r_vis]
    minute[inexact] = 0
    sec = np.where(~inexact & (rng.random(n) < 0.5), rng.integers(1, 60, n), 0)
    date = V["date"][r_vis]
    us = ((date * 1440 + minute) * 60 + sec) * 1_000_000
    hosp = V["hosp"][r_vis]
    person = vp[r_vis]

    # complaints: std null 16.58%
    std_null_v = rng.random(nvis) < 0.1658
    std = np.where(std_null_v, -1, V["concept"])[r_vis]

    # triage priorities (visit level)
    age_v = V["age"]
    lat = -(0.9 * V["urgent"] + 0.5 * V["z"] + 0.02 * (age_v - 54)) + rng.standard_normal(nvis)
    pr_in = np.full(nvis, np.nan)
    nn = np.flatnonzero(rng.random(nvis) >= 0.2169)
    pr_in[nn] = rank_assign(lat[nn], counts_from_shares(len(nn), [.01, .05, .17, .36, .30, .11]))
    pr_out = np.full(nvis, np.nan)
    nn = np.flatnonzero(rng.random(nvis) >= 0.5164)
    pr_out[nn] = rank_assign(lat[nn] + 0.8 * rng.standard_normal(len(nn)),
                             counts_from_shares(len(nn), [.005, .015, .08, .39, .33, .18]))

    # arrival type: ambulance family for the old, frail, urgent and admitted
    at_vals, at_counts, _ = parse_list(LIST_C)
    walk = np.array([v in ARRIVAL_WALK for v in at_vals])
    amb_share = at_counts[~walk].sum() / at_counts.sum()
    at = np.full(nvis, -1)
    nn = np.flatnonzero(rng.random(nvis) >= 0.366)
    sc = 0.04 * (age_v[nn] - 54) + 0.6 * V["z"][nn] + 1.0 * V["urgent"][nn] + 0.8 * (V["cls"][nn] == ADMIT) \
        + rng.standard_normal(len(nn))
    n_amb = int(round(amb_share * len(nn)))
    amb_rows = nn[np.argsort(-sc)[:n_amb]]
    walk_rows = np.setdiff1d(nn, amb_rows)
    amb_ids, walk_ids = np.flatnonzero(~walk), np.flatnonzero(walk)
    at[amb_rows] = amb_ids[choice_p(rng, len(amb_rows), at_counts[amb_ids])]
    at[walk_rows] = walk_ids[choice_p(rng, len(walk_rows), at_counts[walk_ids])]
    konsult = at_vals.index("Konsult (inneliggande SÖS)")
    at[(at == konsult) & (V["hosp"] != SOS)] = at_vals.index("Internt")

    # times
    doc_kn = knots([(0, 0), (0.05, 5), (0.25, 26), (0.5, 60), (0.75, 125), (0.95, 291), (0.99, 600),
                    (0.999, 1440), (1, 2880)])
    tot_kn = knots([(0, 0), (0.05, 33.08), (0.25, 113), (0.5, 191), (0.75, 297), (0.95, 527), (0.99, 900),
                    (0.999, 1800), (1, 2880)])
    # doctor and total times share a latent so that doctor <= total holds naturally
    z1 = rng.standard_normal(nvis)
    z2 = 0.85 * z1 + np.sqrt(1 - 0.85 ** 2) * rng.standard_normal(nvis)
    u_doc = (np.argsort(np.argsort(z1)) + 0.5) / nvis
    u = (np.argsort(np.argsort(z2)) + 0.5) / nvis
    tot = np.where(u < 0.006, 1.0, np.round(inv_cdf(u, tot_kn)))
    frac = (rng.random(nvis) < 0.1) & (tot > 1) & (tot < 2880)
    tot[frac] += rng.integers(1, 60, frac.sum()) / 60
    tot[inexact_v | (rng.random(nvis) < 0.0007)] = np.nan
    doc = np.round(inv_cdf(u_doc, doc_kn))
    doc[inexact_v | (rng.random(nvis) < (0.4094 - 0.0513) / (1 - 0.0513))] = np.nan
    both = ~np.isnan(doc) & ~np.isnan(tot)
    want_bad = int(round(0.005 * both.sum()))
    bad_idx = np.flatnonzero(both & (doc > tot))
    if len(bad_idx) > want_bad:
        fix = bad_idx[rng.permutation(len(bad_idx))[want_bad:]]
        doc[fix] = np.floor(tot[fix] * rng.uniform(0.1, 0.9, len(fix)))
    else:
        good = np.flatnonzero(both & (doc <= tot))
        brk = rng.choice(good, want_bad - len(bad_idx), replace=False)
        doc[brk] = tot[brk] + rng.integers(1, 60, len(brk))
    V["total"] = tot

    # care unit: hospital vocabulary, extra rows of a visit usually move to another unit
    cu_vocab, cu_model = care_unit_model()
    cu = np.full(n, -1)
    for h in range(12):
        rows = np.flatnonzero(hosp == h)
        ids, w = cu_model[h]
        cu[rows] = ids[choice_p(rng, len(rows), w)]
        again = rows[~first[rows]]
        prev_same = cu[again] == cu[again - 1]
        redo = again[prev_same]
        cu[redo] = ids[choice_p(rng, len(redo), w)]

    cls = V["cls"][r_vis]
    obduk_row = V["obduk"][r_vis]
    exit_to, exit_vocab = make_exit_to(rng, n, hosp, cls, obduk_row)
    exit_cu, exit_cu_vocab = make_exit_cu(rng, n, hosp, cls, obduk_row)
    nonstd, nonstd_vocab = make_nonstd(rng, n, C, V["concept"][r_vis], std < 0, hosp)

    minute_vocab = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(60)]
    atime = np.where(inexact, -1, minute)
    age = age_years(P["birth"][person], date)

    t = pa.table([
        str_col(hosp, HOSPITALS), pa.array(P["id"][person]), ts_col(us), pa.array(inexact),
        pa.array(age.astype(float)), i32_col(P["male"][person]), str_col(std, C["terms"]),
        date_col(date), str_col(atime, minute_vocab), f64_col(doc[r_vis]), f64_col(tot[r_vis]),
        f64_col(pr_in[r_vis]), str_col(at[r_vis], at_vals), str_col(cu, cu_vocab), f64_col(pr_out[r_vis]),
        str_col(exit_to, exit_vocab), str_col(exit_cu, exit_cu_vocab), str_col(nonstd, nonstd_vocab),
    ], schema=ED_SCHEMA)
    return t


def make_exit_to(rng, n, hosp, cls, obduk_row):
    vals, counts, _ = parse_list(LIST_D)
    nonnull_real = REAL_ROWS["ed"] * (1 - 0.3223)
    vocab, vclass, vhosp, vw = [], [], [], []
    for v, c in zip(vals, counts):
        vocab.append(v)
        vclass.append(EXIT_CLASS_OF.get(v, ADMIT))
        vhosp.append(exit_hospital(v))
        vw.append(c)
    vclass, vhosp, vw = np.array(vclass), np.array(vhosp), np.array(vw, float)
    tails = {ADMIT: [], HOME: [(s, -1) for s in EXIT_TAIL_HOME], TRANSFER: [(s, -1) for s in EXIT_TAIL_TRANSFER]}
    for h, names in EXIT_TAIL_ADMIT.items():
        tails[ADMIT] += [(s, h) for s in names]
    cap = counts.min() * 0.9
    for c, items in tails.items():
        tail_total = EXIT_CLASS_SHARE[c] * nonnull_real - vw[vclass == c].sum()
        if c == ADMIT:
            tw = np.concatenate([capped_zipf(len(names), tail_total * HOSP_P[h], cap, 0.7)
                                 for h, names in EXIT_TAIL_ADMIT.items()])
        else:
            tw = capped_zipf(len(items), tail_total, cap, 0.7)
        for (s, h), w in zip(items, tw):
            vocab.append(s)
            vclass = np.append(vclass, c)
            vhosp = np.append(vhosp, h)
            vw = np.append(vw, w)

    out = np.full(n, -1)
    nonnull = ~obduk_row & (rng.random(n) >= 0.3223)
    out[~nonnull] = -2  # reserved for null
    for c in range(5):
        rows_c = nonnull & (cls == c)
        n_c = rows_c.sum()
        items = np.flatnonzero(vclass == c)
        share = vw[items] / vw[items].sum()
        spec = items[vhosp[items] >= 0]
        targets = np.zeros(len(vocab), np.int64)
        targets[spec] = np.round(share[vhosp[items] >= 0] * n_c).astype(np.int64)
        key = np.where(rows_c, hosp, -99)
        allocate(rng, out, key, vhosp[spec], spec, targets[spec])
        gen = items[vhosp[items] < 0]
        rows = np.flatnonzero(rows_c & (out == -1))
        out[rows] = gen[choice_p(rng, len(rows), vw[gen])]
    out[out < 0] = -1
    return out, vocab


def make_exit_cu(rng, n, hosp, cls, obduk_row):
    vals, counts, _ = parse_list(LIST_E)
    real = REAL_ROWS["ed"]
    vocab = list(vals) + ["OBDUK"]
    out = np.full(n, -1)
    obduk_id = len(vocab) - 1
    out[obduk_row] = obduk_id
    home = np.flatnonzero((cls == HOME) & ~obduk_row)
    left = np.flatnonzero((cls == LEFT) & ~obduk_row)
    hem, avv = vals.index("Hem"), vals.index("Avvikit")
    out[rng.choice(home, int(round(counts[hem] / real * n)), replace=False)] = hem
    out[rng.choice(left, min(len(left), int(round(counts[avv] / real * n))), replace=False)] = avv
    # admitted and transferred visits get a receiving unit
    unit_ids, unit_h, unit_w = [], [], []
    for i, v in enumerate(vals):
        if v in ("Hem", "Avvikit"):
            continue
        unit_ids.append(i)
        unit_h.append(EXIT_CU_HOSP.get(v, SU if v.isdigit() else -1))
        unit_w.append(counts[i])
    tail_total = (0.0965 * real - counts.sum() - 903)
    cap = counts.min() * 0.9
    for h, names in EXIT_CU_TAIL.items():
        for name, w in zip(names, capped_zipf(len(names), tail_total * 0.85 * HOSP_P[h], cap, 0.8)):
            vocab.append(name)
            unit_ids.append(len(vocab) - 1)
            unit_h.append(h)
            unit_w.append(w)
    for name, w in zip(EXIT_CU_GENERIC_TAIL, capped_zipf(len(EXIT_CU_GENERIC_TAIL), tail_total * 0.15, 6490, 1.2)):
        vocab.append(name)
        unit_ids.append(len(vocab) - 1)
        unit_h.append(-1)
        unit_w.append(w)
    unit_ids, unit_h, unit_w = np.array(unit_ids), np.array(unit_h), np.array(unit_w, float)
    elig = ((cls == ADMIT) | (cls == TRANSFER)) & ~obduk_row
    n_units = int(round(unit_w.sum() / real * n))
    chosen = rng.choice(np.flatnonzero(elig), min(n_units, elig.sum()), replace=False)
    mask = np.zeros(n, bool)
    mask[chosen] = True
    share = unit_w / unit_w.sum()
    spec = np.flatnonzero(unit_h >= 0)
    targets = np.zeros(len(vocab), np.int64)
    targets[unit_ids[spec]] = np.round(share[spec] * len(chosen)).astype(np.int64)
    key = np.where(mask, hosp, -99)
    allocate(rng, out, key, unit_h[spec], unit_ids[spec], targets[unit_ids[spec]])
    gen = np.flatnonzero(unit_h < 0)
    rows = np.flatnonzero(mask & (out == -1))
    out[rows] = unit_ids[gen][choice_p(rng, len(rows), unit_w[gen])]
    return out, vocab


def abbreviate(term):
    words = term.split()
    stem = words[0].upper().replace(",", "")
    return stem[:8]


def make_nonstd(rng, n, C, concept, std_null, hosp):
    vals, counts, _ = parse_list(LIST_F)
    real = REAL_ROWS["ed"]
    vocab = list(vals)
    group_of = C["group"][concept]
    gidx = vocab_index(C["groups"])
    style = ABBREV_STYLE[hosp].astype(np.int64)  # 0 plain, 1 abbrev
    out = np.full(n, -1)
    targets = np.round(counts / real * n).astype(np.int64)
    ids = np.arange(len(vals))
    placeholder = np.array([NONSTD_MAP[v][0] is None for v in vals])
    sty = np.array([{"plain": 0, "abbrev": 1, "any": 2}[NONSTD_MAP[v][1]] for v in vals])
    grp = np.array([gidx[NONSTD_MAP[v][0]] if NONSTD_MAP[v][0] else -1 for v in vals])
    # placeholders take std-null rows first
    key = np.where(std_null, style, -99)
    sel = placeholder & (sty < 2)
    allocate(rng, out, key, sty[sel], ids[sel], targets[sel])
    sel = placeholder & (sty == 2)
    allocate(rng, out, np.where(std_null, 0, -99), np.zeros(sel.sum(), np.int64), ids[sel], targets[sel])
    # concept labels, style-specific before style-agnostic
    sel = ~placeholder & (sty < 2)
    allocate(rng, out, group_of * 2 + style, grp[sel] * 2 + sty[sel], ids[sel], targets[sel])
    sel = ~placeholder & (sty == 2)
    allocate(rng, out, group_of, grp[sel], ids[sel], targets[sel])
    # nulls, then the long tail of local labels
    free = np.flatnonzero(out == -1)
    n_null = int(round(0.091 * n))
    nulls = rng.choice(free, min(n_null, len(free)), replace=False)
    out[nulls] = -2
    rows = np.flatnonzero(out == -1)
    listed = set(vals)
    tail_index = {}
    suffix = choice_p(rng, len(rows), NONSTD_TAIL_SUFFIX_P)
    for r, s in zip(rows, suffix):
        term = C["terms"][concept[r]].split(" [")[0]
        if ABBREV_STYLE[hosp[r]]:
            base = ("TR" if C["dx"][concept[r]].startswith("inj") else "") + abbreviate(term)
        else:
            base = term[0].upper() + term[1:]
        label = base + NONSTD_TAIL_SUFFIX[s]
        if label in listed:
            label = base + "_"
        j = tail_index.get(label)
        if j is None:
            j = tail_index[label] = len(vocab)
            vocab.append(label)
        out[r] = j
    out[out < 0] = -1
    return out, vocab


# =============================================================================
# Section 5: sos_patreg_wide
# =============================================================================

DIAG_RATES = [85.549, 25.438, 9.338, 4.505, 2.405, 1.299, 0.687, 0.363, 0.153, 0.082, 0.044, 0.026, 0.015, 0.008,
              0.004, 0.003, 0.002, 0.0015, 0.001, 0.0008, 0.0006, 0.0005, 0.0004, 0.0003, 0.0003, 0.0002, 0.0002,
              0.0002, 0.0001, 0.0001]
OP_RATES = [35.917, 14.078, 5.707, 2.641, 1.355, 0.729, 0.411, 0.248, 0.157, 0.101, 0.068, 0.050, 0.038, 0.028,
            0.018, 0.014, 0.011, 0.009, 0.007, 0.006, 0.005, 0.004, 0.0035, 0.003, 0.0025, 0.002, 0.0015, 0.001,
            0.001, 0.0008]
OPD1_RATE = 5.830
EKOD_RATES = [6.758, 0.283, 0.054, 0.021, 0.009, 0, 0]

PATREG_SCHEMA = pa.schema(
    [("from", pa.string()), ("source_row", pa.int32()), ("lopnr_swed", pa.float64()), ("INDATUM", pa.date32()),
     ("UTDATUM", pa.date32()), ("Hdia", pa.string())]
    + [(f"DIAGNOS{i}", pa.string()) for i in range(1, 31)]
    + [(f"EKOD{i}", pa.string()) for i in range(1, 8)]
    + [(f"OP{i}", pa.string()) for i in range(1, 31)]
    + [(f"OPD{i}", pa.date32()) for i in range(1, 31)]
    + [("care_level", pa.string()), ("SJUKHUS", pa.string()), ("KTYP", pa.int32()), ("INSATT", pa.string()),
       ("UTSATT", pa.string()), ("DIA_ANT", pa.float64()), ("OP_ANT", pa.float64()), ("is_female", pa.int32()),
       ("SENINV", pa.string()), ("SENUTV", pa.string())]
)


def background_date_knots(fixed_ind, fixed_utd, fixed_inpat, n_total, n_bin, n_bout):
    """Date CDFs for background episodes, chosen so that background plus the fixed (ED-linked,
    terminal, echo) episodes reproduce the INDATUM and UTDATUM quantiles.

    Inpatient background is sampled by discharge date, outpatient background by start date.
    Before the q05 knot the overall INDATUM CDF is drawn as a straight line from the inpatient
    mass at 2001-01-01, because outpatient rows only exist from 2001 on."""
    grid = np.arange(REG_START, REG_END + 1)
    all_kn = knots([(0, "1996-01-01"), (0.05, "2001-10-17"), (0.25, "2005-12-13"), (0.5, "2010-05-04"),
                    (0.75, "2013-11-19"), (0.95, "2016-05-12"), (1, "2016-12-31")])
    in_kn = knots([(0, "1996-10-15"), (0.05, "1998-09-09"), (0.25, "2004-06-08"), (0.5, "2009-04-09"),
                   (0.75, "2013-03-15"), (0.95, "2016-03-18"), (1, "2016-12-31")])

    def cum_counts(days):
        c = np.bincount(np.clip(days - REG_START, 0, len(grid) - 1), minlength=len(grid))
        return np.cumsum(c).astype(float)

    def as_cdf(f):
        f = np.maximum.accumulate(np.clip(f, 0, None))
        f = f / f[-1] + np.arange(len(f)) * 1e-12
        return f / f[-1], grid.astype(float)

    n_fix_in = fixed_inpat.sum()
    g_in = (cdf(grid, in_kn) * (n_fix_in + n_bin) - cum_counts(fixed_utd[fixed_inpat])) / max(n_bin, 1)
    g_in = as_cdf(g_in)

    target = cdf(grid, all_kn)
    q05 = D("2001-10-17")
    before = grid < OUTPAT_START
    inpat_mass = (cum_counts(fixed_ind)[OUTPAT_START - REG_START] + n_bin * g_in[0][OUTPAT_START - REG_START]) / n_total
    ramp = (grid >= OUTPAT_START) & (grid < q05)
    target[ramp] = inpat_mass + (0.05 - inpat_mass) * (grid[ramp] - OUTPAT_START) / (q05 - OUTPAT_START)
    g_out = (target * n_total - cum_counts(fixed_ind) - n_bin * g_in[0]) / max(n_bout, 1)
    g_out[before] = 0
    g_out = g_out - g_out[OUTPAT_START - REG_START]
    g_out[before] = 0
    return as_cdf(g_out), g_in


def make_patreg_tables(rng, P, V, DC, pools):
    """Build the episode skeleton, then fill codes. Returns a pyarrow table."""
    n = P["n"]
    death = P["death"]
    birth = P["birth"]
    female = ~P["male"]

    # --- ED-linked episodes
    ok = ~V["err"] & (V["date"] <= REG_END)
    adm = ok & (V["cls"] == ADMIT)
    outp = ok & ~adm & (rng.random(V["n"]) < OUTPAT_FROM_ED)
    ei = np.flatnonzero(adm)
    e_los = draw_los(rng, len(ei))
    e_ind = V["date"][ei]
    e_utd = e_ind + e_los
    dp = death[V["person"][ei]]
    term = V["inhosp_near"][ei]
    e_utd = np.where(term, dp, np.minimum(e_utd, np.maximum(dp - 1, e_ind)))
    keep = term | (e_utd <= REG_END)
    ei, e_ind, e_utd, term = ei[keep], e_ind[keep], e_utd[keep], term[keep]
    oi = np.flatnonzero(outp)

    rows = dict(person=[], ind=[], utd=[], inpat=[], source=[], visit=[], terminal=[])

    def add(person, ind, utd, inpat, source, visit=None, terminal=None):
        k = len(person)
        rows["person"].append(np.asarray(person, np.int64))
        rows["ind"].append(np.asarray(ind, np.int64))
        rows["utd"].append(np.asarray(utd, np.int64))
        rows["inpat"].append(np.full(k, inpat))
        rows["source"].append(np.full(k, source))
        rows["visit"].append(np.full(k, -1) if visit is None else np.asarray(visit))
        rows["terminal"].append(np.zeros(k, bool) if terminal is None else np.asarray(terminal))

    SRC_ED, SRC_TERM, SRC_BG, SRC_ERR, SRC_ECHO = range(5)
    add(V["person"][ei], e_ind, e_utd, True, SRC_ED, ei, term)
    add(V["person"][oi], V["date"][oi], np.full(len(oi), -1), False, SRC_ED, oi)

    # --- in-hospital deaths without an ED-linked stay
    inhosp = V["p_inhosp"].copy()
    inhosp[V["person"][ei[term]]] = False
    tp = np.flatnonzero(inhosp)
    t_los = np.maximum(draw_los(rng, len(tp)), 1)
    t_ind = np.maximum(death[tp] - t_los, np.maximum(REG_START, birth[tp]))
    add(tp, t_ind, death[tp], True, SRC_TERM, None, np.ones(len(tp), bool))

    # --- register history echoing the cause of death
    ep = DC["echo_person"]
    e_date = death[ep] - 30 - np.floor(rng.random(len(ep)) * 1000).astype(np.int64)
    e_date = np.clip(e_date, np.maximum(REG_START, birth[ep]), np.minimum(REG_END, death[ep] - 1))
    e_in = e_date < OUTPAT_START
    add(ep[e_in], e_date[e_in], e_date[e_in] + draw_los(rng, e_in.sum()), True, SRC_ECHO)
    add(ep[~e_in], e_date[~e_in], np.full((~e_in).sum(), -1), False, SRC_ECHO)

    # --- deliberate error: an episode 1-30 days after death
    dead_exact = np.flatnonzero(P["die"] & (P["inexact"] == 0) & (death <= REG_END - 30))
    n_err = max(1, int(round(ERR_EPISODE_AFTER_DEATH * P["die"].sum())))
    xp = rng.choice(dead_exact, min(n_err, len(dead_exact)), replace=False)
    add(xp, death[xp] + 1 + np.floor(rng.random(len(xp)) * 30).astype(np.int64), np.full(len(xp), -1), False,
        SRC_ERR)

    fixed_person = np.concatenate(rows["person"])
    fixed_inpat = np.concatenate(rows["inpat"])
    total = ROWS_PER_PERSON["patreg"] * n
    t_in = INPAT_SHARE * total * 1.004 - fixed_inpat.sum()  # stays ending after 2016 are dropped
    t_out = (1 - INPAT_SHARE) * total - (~fixed_inpat).sum()

    # --- background episodes
    end = np.minimum(REG_END, death - 1)
    in_lo = np.maximum(UTDATUM_MIN, birth)
    out_lo = np.maximum(OUTPAT_START, birth)
    out_hi = np.minimum(REG_END, death)
    out_kn, in_kn = background_date_knots(np.concatenate(rows["ind"]), np.concatenate(rows["utd"]),
                                          fixed_inpat, total, t_in, t_out)
    expo_in = np.clip(cdf(end + 1.0, in_kn) - cdf(in_lo, in_kn), 0, None)
    expo_out = np.clip(cdf(out_hi + 1.0, out_kn) - cdf(out_lo, out_kn), 0, None)
    amid = ((np.maximum(REG_START, birth) + end) / 2 - birth) / 365.25
    b_in = expo_in * np.exp(0.6 * P["z"] + 0.03 * (amid - 50))
    b_out = expo_out * np.exp(0.4 * P["z"] + 0.012 * (amid - 50))
    fixed_f = female[fixed_person].sum()

    def female_rows(f):
        w_in = b_in * np.where(female, f, 1)
        w_out = b_out * np.where(female, f, 1)
        return fixed_f + t_in * w_in[female].sum() / w_in.sum() + t_out * w_out[female].sum() / w_out.sum()

    target_f = PATREG_FEMALE_ROW_SHARE * (len(fixed_person) + t_in + t_out)
    f = bisect(lambda f: female_rows(f) - target_f, 0.2, 5)
    w_in = b_in * np.where(female, f, 1)
    w_out = b_out * np.where(female, f, 1)
    c_in = draw_nb(rng, t_in * w_in / w_in.sum(), 1.0)
    c_out = draw_nb(rng, t_out * w_out / w_out.sum(), 1.0)

    # sample within each person's window, then quantile-map the pooled dates onto the target
    # distribution: high-rate persons die early, which would otherwise pull mass to the early years
    bp = np.repeat(np.arange(n), c_in)
    utd = quantile_map(rng, qsample_days(rng, len(bp), in_kn, in_lo[bp], end[bp]), in_kn, in_lo[bp], end[bp])
    los = draw_los(rng, len(bp))
    ind = np.maximum(utd - los, np.maximum(REG_START, birth[bp]))
    add(bp, ind, utd, True, SRC_BG)
    bp = np.repeat(np.arange(n), c_out)
    ind = quantile_map(rng, qsample_days(rng, len(bp), out_kn, out_lo[bp], out_hi[bp]), out_kn, out_lo[bp],
                       out_hi[bp])
    add(bp, ind, np.full(len(bp), -1), False, SRC_BG)

    R = {k: np.concatenate(v) for k, v in rows.items()}
    order = np.lexsort((R["ind"], R["person"]))
    R = {k: v[order] for k, v in R.items()}
    nr = len(R["person"])
    log(f"patient register skeleton: {nr:,} episodes")
    return fill_patreg(rng, P, V, DC, pools, R, SRC_ED, SRC_TERM, SRC_ECHO)


def fill_patreg(rng, P, V, DC, pools, R, SRC_ED, SRC_TERM, SRC_ECHO):
    nr = len(R["person"])
    person = R["person"]
    inpat = R["inpat"]
    female = ~P["male"][person]
    age = age_years(P["birth"][person], R["ind"])
    group = np.where(~female, 0, np.where((age >= 15) & (age <= 50), 1, 2))
    icd_vocab = pools["icd_vocab"]
    icd_idx = vocab_index(icd_vocab)
    allowed = pools["icd_allowed"]
    # ED-linked, terminal and echo episodes have a known main diagnosis
    forced = (R["source"] == SRC_ED) | (R["source"] == SRC_TERM) | (R["source"] == SRC_ECHO)

    # --- number of diagnoses: ranked latent, forced rows never empty
    late = R["ind"] >= np.quantile(R["ind"], DIA_ANT_NULL)
    lat = 1.5 * inpat + 0.4 * P["z"][person] + 0.015 * (age - 50) + 0.7 * late + rng.standard_normal(nr)
    lat[forced] += 50
    rates = np.array(DIAG_RATES) / 100
    k_counts = counts_from_shares(nr, np.concatenate([[1 - rates[0]], -np.diff(rates), [rates[-1]]]))
    kdiag = rank_assign(lat, k_counts)

    # --- main diagnosis
    hdia = np.full(nr, -1)
    ed_rows = np.flatnonzero(R["source"] == SRC_ED)
    dx_keys = [V_DX_KEY[i] for i in V["concept"][R["visit"][ed_rows]]] if len(ed_rows) else []
    hdia[ed_rows] = map_dx(rng, dx_keys, icd_idx, group[ed_rows], pools)
    def reg_code(p, default):
        code = DC["code_of_person"].get(int(p))
        if code is None:
            return default
        c = "T14" if is_external(code) else register_format(code)
        return icd_idx.get(c, icd_idx.get(c[:3], icd_idx["R50-R69"]))

    term_rows = np.flatnonzero(R["source"] == SRC_TERM)
    hdia[term_rows] = [reg_code(p, icd_idx["R50-R69"]) for p in person[term_rows]]
    echo_rows = np.flatnonzero(R["source"] == SRC_ECHO)
    hdia[echo_rows] = [icd_idx[DC["echo_code"][p]] for p in person[echo_rows]]
    # ED-linked inpatient stays that end in death take the death cause too
    ed_term = np.flatnonzero(R["terminal"] & (R["source"] == SRC_ED))
    hdia[ed_term] = [reg_code(p, hdia[r]) for p, r in zip(person[ed_term], ed_term)]
    rest = np.flatnonzero((kdiag >= 1) & (hdia < 0))
    n_nonnull = (kdiag >= 1).sum()
    target = pools["g_weights"] / pools["g_weights"].sum() * n_nonnull
    fixed_counts = np.bincount(hdia[hdia >= 0], minlength=len(icd_vocab))
    resid = np.clip(target - fixed_counts, 0, None)
    hdia[rest] = sample_grouped(rng, group[rest], resid / resid.sum(), allowed)
    hdia[kdiag == 0] = -1
    diag1 = hdia.copy()
    cand = np.flatnonzero((kdiag >= 1) & ~forced)
    hdia[rng.choice(cand, int(round(0.00046 * nr)), replace=False)] = -1

    diag_pos = {1: (np.flatnonzero(kdiag >= 1), diag1[kdiag >= 1])}
    for k in range(2, 31):
        idx = np.flatnonzero(kdiag >= k)
        w = pools["h_weights"] if k == 2 else (pools["i_weights"] if k < 7 else pools["i7_weights"])
        vals = sample_grouped(rng, group[idx], w / w.sum(), allowed)
        for j in range(1, k):  # resample duplicates once
            pidx, pv = diag_pos[j]
            dup = vals == pv[np.searchsorted(pidx, idx)]
            if dup.any():
                vals[dup] = sample_grouped(rng, group[idx[dup]], w / w.sum(), allowed)
        diag_pos[k] = (idx, vals)

    # --- external causes for injuries
    st = np.zeros(nr, bool)
    has = diag1 >= 0
    st[has] = np.array([icd_vocab[c][0] in "ST" and "-" not in icd_vocab[c] for c in diag1[has]])
    target_e = EKOD_RATES[0] / 100 * nr
    p_st = EKOD_ST_SHARE if EKOD_ST_SHARE * st.sum() <= target_e else target_e / max(st.sum(), 1)
    p_other = max(target_e - p_st * st.sum(), 0) / max((~st).sum(), 1)
    e1 = rng.random(nr) < np.where(st, p_st, p_other)
    ek_vocab = pools["ekod_vocab"]
    ekod_pos = {}
    have = e1
    for k in range(1, 8):
        idx = np.flatnonzero(have)
        ekod_pos[k] = (idx, choice_p(rng, len(idx), pools["ekod_weights"]))
        if k < 7:
            cond = EKOD_RATES[k] / EKOD_RATES[k - 1] if EKOD_RATES[k - 1] > 0 else 0
            nxt = np.zeros(nr, bool)
            nxt[idx[rng.random(len(idx)) < cond]] = True
            have = nxt

    # --- procedures
    lat = 1.6 * inpat + 0.3 * P["z"][person] + rng.standard_normal(nr)
    rates = np.array(OP_RATES) / 100
    kop = rank_assign(lat, counts_from_shares(nr, np.concatenate([[1 - rates[0]], -np.diff(rates), [rates[-1]]])))
    has_op = kop >= 1
    dated = has_op & inpat
    target_d = int(round(OPD1_RATE / 100 * nr))
    if dated.sum() > target_d:
        idx = np.flatnonzero(dated)
        dated[rng.choice(idx, dated.sum() - target_d, replace=False)] = False
    else:
        idx = np.flatnonzero(has_op & ~inpat)
        dated[rng.choice(idx, min(len(idx), target_d - dated.sum()), replace=False)] = True
    op_pos, opd_pos = {}, {}
    utd = R["utd"]
    for k in range(1, 31):
        idx = np.flatnonzero(kop >= k)
        w = pools["k_weights"] if k == 1 else pools["l_weights"]
        op_pos[k] = (idx, choice_p(rng, len(idx), w))
        didx = idx[dated[idx]]
        span = np.where(inpat[didx], utd[didx] - R["ind"][didx], 0)
        dd = R["ind"][didx] + np.floor(rng.random(len(didx)) * (span + 1)).astype(np.int64)
        spike = rng.random(len(didx)) < 0.003
        dd[spike] = D("2012-06-01")
        opd_pos[k] = (didx, dd)

    # --- other columns
    sj_vocab = pools["sj_vocab"]
    sj_idx = vocab_index(sj_vocab)
    sj = np.full(nr, -1)
    vis = R["visit"]
    ed_any = vis >= 0
    vh = V["hosp"][vis[ed_any]]
    sj_ed = np.empty(ed_any.sum(), np.int64)
    for h in range(12):
        m = vh == h
        codes = HOSP_SJUKHUS[h]
        sj_ed[m] = np.array([sj_idx[c] for c in codes])[rng.integers(0, len(codes), m.sum())]
    sj[ed_any] = sj_ed
    term = np.flatnonzero(R["source"] == SRC_TERM)
    sj[term] = [sj_idx[HOSP_SJUKHUS[P["home"][p]][0]] for p in person[term]]
    rest = np.flatnonzero(sj < 0)
    target = pools["m_weights"] / pools["m_weights"].sum() * nr
    resid = np.clip(target - np.bincount(sj[sj >= 0], minlength=len(sj_vocab)), 0, None)
    sj[rest] = choice_p(rng, len(rest), resid)
    sj[rng.choice(rest, int(round(0.0004 * nr)), replace=False)] = -1

    ktyp = np.full(nr, -1)
    outp = np.flatnonzero(~inpat)
    p_null = (0.614 - inpat.mean()) / max((~inpat).mean(), 1e-9)
    kk = outp[rng.random(len(outp)) >= p_null]
    ktyp[kk] = exact_draw(rng, len(kk), [.85, .03, .02, .01, .01, .005, .005, .01, .04, .02])

    ins = np.full(nr, -1)
    inp = np.flatnonzero(inpat)
    ins[inp] = choice_p(rng, len(inp), [0.121, 0.008, 0.871])  # '1','2','3'
    ins[inp[rng.random(len(inp)) < 0.0039]] = -1
    uts = np.full(nr, -1)
    uts[inp] = choice_p(rng, len(inp), [0.114, 0.026, 0.846, 0.000002])  # '1','2','3','5'
    uts[uts == 3] = 4
    uts[inp[rng.random(len(inp)) < 0.0023]] = -1
    uts[R["terminal"]] = 3  # '4' = died
    ins_vocab = ["1", "2", "3"]
    uts_vocab = ["1", "2", "3", "4", "5"]

    dia_ant = np.where(late, kdiag.astype(float), np.nan)
    big = late & (kdiag == 30)
    dia_ant[big] = 30 + rng.integers(0, 36, big.sum())
    op_ant = np.full(nr, np.nan)
    opr = np.flatnonzero(has_op)
    cut = np.quantile(R["ind"][opr], 1 - OP_ANT_SHARE)
    sel = opr[R["ind"][opr] >= cut]
    op_ant[sel] = kop[sel]
    if nr > 1_000_000 and len(sel):
        op_ant[sel[rng.integers(0, len(sel))]] = 603

    sen_inv, sen_utv = migration_strings(rng, P, person)

    cols = [str_col(np.zeros(nr, np.int64), ["patreg"]),
            i32_col(rng.permutation(nr) + 1), pa.array(P["id"][person]),
            date_col(R["ind"]), date_col(np.where(inpat, R["utd"], 0), mask=~inpat),
            str_col(hdia, icd_vocab)]
    for k in range(1, 31):
        full = np.full(nr, -1)
        idx, v = diag_pos[k]
        full[idx] = v
        cols.append(str_col(full, icd_vocab))
    for k in range(1, 8):
        full = np.full(nr, -1)
        if EKOD_RATES[k - 1] > 0:
            idx, v = ekod_pos[k]
            full[idx] = v
        cols.append(str_col(full, ek_vocab))
    for k in range(1, 31):
        full = np.full(nr, -1)
        idx, v = op_pos[k]
        full[idx] = v
        cols.append(str_col(full, pools["kva_vocab"]))
    for k in range(1, 31):
        full = np.zeros(nr, np.int64)
        mask = np.ones(nr, bool)
        idx, v = opd_pos[k]
        full[idx] = v
        mask[idx] = False
        cols.append(date_col(full, mask=mask))
    cols += [str_col(np.where(inpat, 1, 0), ["spec_outpatient", "inpatient"]), str_col(sj, sj_vocab),
             i32_col(np.maximum(ktyp, 0), mask=ktyp < 0), str_col(ins, ins_vocab), str_col(uts, uts_vocab),
             f64_col(dia_ant), f64_col(op_ant), i32_col(female), sen_inv, sen_utv]
    return pa.Table.from_arrays(cols, schema=PATREG_SCHEMA)


V_DX_KEY: list = []


def map_dx(rng, keys, icd_idx, group, pools):
    keys = np.asarray(keys, dtype=object)
    out = np.full(len(keys), -1)
    for key in sorted(set(keys)):
        rows = np.flatnonzero(keys == key)
        if key.startswith("inj:"):
            codes = key[4:].split(",")
            probs = np.array([0.6] + [0.4 / (len(codes) - 1)] * (len(codes) - 1)) if len(codes) > 1 else np.ones(1)
        elif key in DX_MAP:
            codes = [c for c, _ in DX_MAP[key]]
            probs = np.array([p for _, p in DX_MAP[key]])
        else:
            continue
        cidx = np.array([icd_idx[c] for c in codes])
        ok = pools["icd_allowed"][:, cidx]
        for g in range(3):
            rg = rows[group[rows] == g]
            if len(rg) == 0:
                continue
            p = probs * ok[g]
            if p.sum() == 0:
                continue
            out[rg] = cidx[choice_p(rng, len(rg), p)]
    rest = np.flatnonzero(out < 0)
    if len(rest):
        out[rest] = sample_grouped(rng, group[rest], pools["g_weights"] / pools["g_weights"].sum(),
                                   pools["icd_allowed"])
    return out


def migration_strings(rng, P, person):
    """SENINV / SENUTV: person-level strings repeated on all the person's rows."""
    n = P["n"]
    rows_per = np.bincount(person, minlength=n).astype(float)
    nr = len(person)
    foreign = P["foreign"]
    birth, death = P["birth"], P["death"]

    def pick(target_rows, p_native_rel):
        # probability for foreign-born persons; natives get p_native_rel times that
        rf = rows_per[foreign].sum()
        rn = rows_per[~foreign].sum()
        p_f = min(1.0, target_rows / (rf + p_native_rel * rn))
        return rng.random(n) < np.where(foreign, p_f, p_f * p_native_rel)

    def fmt(days, yyyymm):
        y, m, d = ymd(days)
        return [f"{a:04d}{b:02d}" if s else f"{a:04d}{b:02d}{c:02d}" for a, b, c, s in zip(y, m, d, yyyymm)]

    inv = pick((1 - 0.7901) * nr, 0.03)
    lo = np.maximum(birth, D("1945-01-01"))
    hi = np.minimum(REG_END, death)
    days = lo + np.floor(rng.random(n) * np.maximum(hi - lo, 1)).astype(np.int64)
    s_inv = np.array(fmt(days, rng.random(n) < SENINV_YYYYMM_SHARE), dtype=object)
    s_inv[rng.random(n) < SENINV_PLACEHOLDER] = PLACEHOLDER_SPACES
    s_inv[~inv] = None

    utv = pick((1 - 0.9602) * nr, 0.25)
    lo = np.maximum(birth + 365, D("1950-01-01"))
    days = lo + np.floor(rng.random(n) * np.maximum(hi - lo, 1)).astype(np.int64)
    c65 = (rng.random(n) < SENUTV_1965_SHARE) & (birth < D("1964-01-01"))
    days[c65] = D("1965-05-20") + rng.integers(0, 42, c65.sum())
    s_utv = np.array(fmt(days, rng.random(n) < SENINV_YYYYMM_SHARE), dtype=object)
    s_utv[rng.random(n) < SENUTV_PLACEHOLDER] = PLACEHOLDER_SPACES
    s_utv[~utv] = None
    return obj_col(s_inv[person]), obj_col(s_utv[person])


def build_pools(rng):
    pools = {}
    icd = build_icd_pool()
    g_vals, g_counts, _ = parse_list(LIST_G)
    h_vals, h_counts, _ = parse_list(LIST_H)
    i_vals, i_counts, _ = parse_list(LIST_I)
    extra = set(g_vals) | set(h_vals) | set(i_vals)
    for key, lst in DX_MAP.items():
        extra |= {c for c, _ in lst}
    for key in COMPLAINT_INFO.values():
        if key[3].startswith("inj:"):
            extra |= set(key[3][4:].split(","))
    for t in COMPLAINT_TAIL:
        if t[5].startswith("inj:"):
            extra |= set(t[5][4:].split(","))
    vocab = list(dict.fromkeys(icd + sorted(extra)))
    real = REAL_ROWS["patreg"]

    def weights(vals, counts, nonnull, s, boost=None):
        v, w = with_tail(rng, vals, counts, real, nonnull, vocab, s)
        out = np.zeros(len(vocab))
        idx = vocab_index(vocab)
        for a, b in zip(v, w):
            out[idx[a]] += b
        if boost:
            for c, f in boost.items():
                out[idx[c]] *= f
        return out

    pools["icd_vocab"] = vocab
    pools["icd_allowed"] = np.array([icd_sex_rule(c) for c in vocab]).T
    pools["g_weights"] = weights(g_vals, g_counts, 1 - 0.1450, 0.9)
    pools["h_weights"] = weights(h_vals, h_counts, 0.25438, 0.9)
    pools["i_weights"] = weights(i_vals, i_counts, 0.04505, 0.9)
    pools["i7_weights"] = weights(i_vals, i_counts, 0.04505, 0.9,
                                  boost={"Z95": 2.5, "Z92": 2.5, "Z86": 8, "I109": 1.5, "Z85": 2.5})
    j_vals, j_counts, _ = parse_list(LIST_J)
    v, w = with_tail(rng, j_vals, j_counts, real, 0.06758, build_ekod_pool(), 1.0)
    pools["ekod_vocab"], pools["ekod_weights"] = v, w
    kva = build_kva_pool(rng)
    k_vals, k_counts, _ = parse_list(LIST_K)
    l_vals, l_counts, _ = parse_list(LIST_L)
    kv = list(dict.fromkeys(list(k_vals) + list(l_vals) + kva))
    kidx = vocab_index(kv)

    def kweights(vals, counts, nonnull):
        v, w = with_tail(rng, vals, counts, real, nonnull, kv, 1.0)
        out = np.zeros(len(kv))
        for a, b in zip(v, w):
            out[kidx[a]] += b
        return out

    pools["kva_vocab"] = kv
    pools["k_weights"] = kweights(k_vals, k_counts, 0.35917)
    pools["l_weights"] = kweights(l_vals, l_counts, 0.14078)
    m_vals, m_counts, _ = parse_list(LIST_M)
    sj_tail = build_sjukhus_pool(rng, set(m_vals))
    v, w = with_tail(rng, m_vals, m_counts, real, 1 - 0.0004, sj_tail, 1.0)
    pools["sj_vocab"], pools["m_weights"] = v, w
    return pools


# =============================================================================
# Section 6: sos_deathreg_wide
# =============================================================================

MORSAK_RATES = [89.25, 76.04, 58.53, 41.27, 26.77, 16.46, 9.72, 5.61, 3.18, 1.70, 0.88, 0.45, 0.23, 0.12, 0.06,
                0.02, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001] + [0.0] * 26

DEATH_SCHEMA = pa.schema(
    [("from", pa.string()), ("source_row", pa.int32()), ("lopnr_swed", pa.float64()),
     ("inexact_death_date", pa.bool_()), ("date_death_meanimp", pa.date32()), ("date_death_minimp", pa.date32()),
     ("ULORSAK", pa.string())]
    + [(f"MORSAK{i}", pa.string()) for i in range(1, 49)]
    + [("KAP19", pa.string()), ("LK", pa.string()), ("DODSPL", pa.int32()), ("DBGRUND1", pa.int32()),
       ("is_female", pa.int32())]
)


def death_pools(rng):
    pool = build_death_pool()
    real = REAL_ROWS["death"]
    n_vals, n_counts, _ = parse_list(LIST_N)
    coded = 1 - ULORSAK_NULL
    ext_total = 0.0506
    ext_listed = sum(c for v, c in zip(n_vals, n_counts) if is_external(v)) / real
    non_listed = sum(c for v, c in zip(n_vals, n_counts) if not is_external(v)) / real
    vocab = list(dict.fromkeys(list(n_vals) + pool + DEATH_EXT_POOL))
    for lst in (LIST_O, LIST_P, LIST_Q):
        vocab += [v for v in parse_list(lst)[0] if v not in vocab]
    vocab += ["Z924"]
    idx = vocab_index(vocab)
    w = np.zeros(len(vocab))
    for v, c in zip(n_vals, n_counts):
        w[idx[v]] = c / real
    cap = n_counts.min() / real * 0.9
    tail_non = [c for c in pool if c not in set(n_vals)]
    tail_non = [tail_non[i] for i in rng.permutation(len(tail_non))]
    for c, x in zip(tail_non, capped_zipf(len(tail_non), coded - ext_total - non_listed, cap, 0.9)):
        w[idx[c]] += x
    tail_ext = [c for c in DEATH_EXT_POOL if c not in set(n_vals)]
    tail_ext = [tail_ext[i] for i in rng.permutation(len(tail_ext))]
    for c, x in zip(tail_ext, capped_zipf(len(tail_ext), ext_total - ext_listed, cap, 0.9)):
        w[idx[c]] += x

    def pos_weights(text, nonnull, boost=None):
        vals, counts, _ = parse_list(text)
        out = np.zeros(len(vocab))
        for v, c in zip(vals, counts):
            out[idx[v]] = c / real
        tail = [c for c in pool if c not in set(vals)]
        tail = [tail[i] for i in rng.permutation(len(tail))]
        for c, x in zip(tail, capped_zipf(len(tail), nonnull - out.sum(), counts.min() / real * 0.9, 0.9)):
            out[idx[c]] += x
        if boost:
            for c, f in boost.items():
                out[idx[c]] += f
        return out

    allowed = np.array([death_sex_rule(c) for c in vocab]).T
    r_vals, r_counts, _ = parse_list(LIST_R)
    kv, kw = with_tail(rng, r_vals, r_counts, real, 0.0506, KAP19_POOL, 1.0)
    return dict(vocab=vocab, ul=w, allowed=allowed,
                m1=pos_weights(LIST_O, 0.8925), m2=pos_weights(LIST_P, 0.7604), m3=pos_weights(LIST_Q, 0.5853),
                m6=pos_weights(LIST_Q, 0.5853, boost={"X590": 0.01, "Z924": 0.01}),
                kap_vocab=kv, kap_w=kw)


def assign_death_causes(rng, P, DP):
    """Underlying cause per deceased person, decided before the register so the register can echo it."""
    di = np.flatnonzero(P["die"])
    death = P["death"][di]
    vocab = DP["vocab"]
    n = len(di)
    # no cause coded: concentrated in the last year
    n_null = int(round(ULORSAK_NULL * n))
    recent = np.argsort(-(death + rng.integers(0, 200, n)))
    uncoded = np.zeros(n, bool)
    uncoded[recent[:n_null]] = True
    sex = (~P["male"][di]).astype(np.int64)
    ul = np.full(n, -1)
    c = np.flatnonzero(~uncoded)
    ul[c] = sample_grouped(rng, sex[c], DP["ul"] / DP["ul"].sum(), DP["allowed"])
    code_of_person = {int(p): vocab[u] for p, u in zip(di, ul) if u >= 0}
    # register echoes: cancer block before cancer death, heart failure before I50 death
    echo_person, echo_code = [], {}
    for p, u in zip(di, ul):
        if u < 0 or P["death"][p] > REG_END + 365:
            continue
        code = vocab[u]
        if code.startswith("C") and rng.random() < CANCER_ECHO:
            echo_person.append(p)
            echo_code[int(p)] = block_of_c(code[:3])
        elif code.startswith("I50") and rng.random() < HF_ECHO:
            echo_person.append(p)
            echo_code[int(p)] = "I509" if rng.random() < 0.7 else "I500"
    return dict(persons=di, ul=ul, code_of_person=code_of_person,
                echo_person=np.array(echo_person, np.int64), echo_code=echo_code)


def make_death(rng, P, V, DC, DP):
    di = DC["persons"]
    n = len(di)
    vocab = DP["vocab"]
    death = P["death"][di]
    inexact = P["inexact"][di]
    y, m, _ = ymd(death)
    mean_imp = np.where(inexact == 1, from_ymd(y, m, 15), np.where(inexact == 2, from_ymd(y, 6, 15), death))
    min_imp = np.where(inexact == 1, from_ymd(y, m, 1), np.where(inexact == 2, from_ymd(y, 1, 1), death))
    ul = DC["ul"]
    coded = ul >= 0
    sex = (~P["male"][di]).astype(np.int64)

    # contributing causes
    rates = np.array(MORSAK_RATES) / 100
    k_counts = counts_from_shares(coded.sum(), np.concatenate(
        [[1 - rates[0] / (1 - ULORSAK_NULL)], -np.diff(rates) / (1 - ULORSAK_NULL), [rates[-1]]]))
    k = np.zeros(n, np.int64)
    ci = np.flatnonzero(coded)
    k[ci] = rank_assign(rng.random(len(ci)), k_counts)
    ms = np.full((48, n), -1)
    for pos in range(48):
        rows = np.flatnonzero(k > pos)
        if len(rows) == 0:
            continue
        w = DP["m1"] if pos == 0 else DP["m2"] if pos == 1 else DP["m3"] if pos < 5 else DP["m6"]
        vals = sample_grouped(rng, sex[rows], w / w.sum(), DP["allowed"])
        dup = (vals == ul[rows]) | (ms[:pos, rows] == vals).any(axis=0)
        if dup.any():
            vals[dup] = sample_grouped(rng, sex[rows[dup]], w / w.sum(), DP["allowed"])
        ms[pos, rows] = vals

    # injury code for external causes
    kap = np.full(n, -1)
    ext = np.array([u >= 0 and is_external(vocab[u]) for u in ul])
    kap[ext] = choice_p(rng, ext.sum(), DP["kap_w"])

    # place of death
    nulls = rng.choice(n, int(round(0.18 * n)), replace=False)
    forced = V["p_inhosp"][di] | V["p_obduk"][di]
    forced[nulls] = False
    dodspl = np.full(n, -1)
    dodspl[forced] = 1
    rest = np.setdiff1d(np.flatnonzero(~forced), nulls)
    nn = n - len(nulls)
    target = np.array([.45, .42, .11, .02]) * nn
    target[0] -= forced.sum()
    dodspl[rest] = 1 + exact_draw(rng, len(rest), np.clip(target, 0, None))
    dbg = np.full(n, -1)
    nn = np.flatnonzero(rng.random(n) >= 0.8994)
    dbg[nn] = 1 + exact_draw(rng, len(nn), [.47, .02, .49, .02])

    lk_vocab, lk = municipality(rng, P, di)

    cols = [str_col(np.zeros(n, np.int64), ["deathreg"]), i32_col(rng.permutation(n) + 1), pa.array(P["id"][di]),
            pa.array(inexact > 0), date_col(mean_imp), date_col(min_imp), str_col(ul, vocab)]
    cols += [str_col(ms[pos], vocab) for pos in range(48)]
    cols += [str_col(kap, DP["kap_vocab"]), str_col(lk, lk_vocab), i32_col(np.maximum(dodspl, 0), mask=dodspl < 0),
             i32_col(np.maximum(dbg, 0), mask=dbg < 0), i32_col(sex)]
    return pa.Table.from_arrays(cols, schema=DEATH_SCHEMA)


def municipality(rng, P, di):
    s_vals, s_counts, _ = parse_list(LIST_S)
    real = REAL_ROWS["death"]
    listed = dict(zip(s_vals, s_counts / real))
    vocab = []
    region_codes = {}
    for h in range(12):
        r = HOSP_REGION[h]
        if r in region_codes or HOSP_LK[r] is None:
            continue
        codes = HOSP_LK[r].split()
        share = HOSP_P[[i for i in range(12) if HOSP_REGION[i] == r]].sum()
        w = np.array([listed.get(c, 0.0) for c in codes])
        tail = np.flatnonzero(w == 0)
        if share > w.sum():
            w[tail] = capped_zipf(len(tail), share - w.sum(), min(s_counts) / real * 0.9, 1.0)
        start = len(vocab)
        vocab += codes
        region_codes[r] = (np.arange(start, start + len(codes)), w)
    lk = np.full(len(di), -1)
    reg = np.array([HOSP_REGION[h] for h in P["home"][di]])
    for r, (ids, w) in region_codes.items():
        rows = np.flatnonzero(reg == r)
        lk[rows] = ids[choice_p(rng, len(rows), w)]
    lk[rng.random(len(di)) < 0.0001] = -1
    return vocab, lk


# =============================================================================
# Section 7: ehr_danderyd_lab_meas
# =============================================================================

LAB_SCHEMA = pa.schema([
    ("from", pa.string()), ("lopnr_swed", pa.float64()), ("inexact_time", pa.bool_()),
    ("datetime_analysis", pa.timestamp("us", tz="UTC")), ("term_name", pa.string()), ("term_id", pa.float64()),
    ("value_str", pa.string()), ("value_num", pa.float64()), ("unit", pa.string()), ("lt_flag", pa.bool_()),
    ("gt_flag", pa.bool_()), ("other_flag", pa.string()), ("ref_val1", pa.string()),
])


def lab_catalogue():
    terms = [dict(name=t[0], id=t[1], unit=t[2], ref=(t[3], t[4]), dist=t[5], kind=t[6], w=0.0) for t in MAIN_TERMS]
    others = [dict(name=t[0], id=None, unit=t[1], ref=(t[2], t[3]), dist=t[5], kind=t[6], w=float(t[4]))
              for t in OTHER_TERMS]
    other_total = sum(o["w"] for o in others) / (1 - NEWS_SHARE_OF_OTHER)
    news = dict(name=NEWS_TERM[0], id=6520, unit=None, ref=(None, None), dist=NEWS_TERM[5], kind="obs",
                w=NEWS_SHARE_OF_OTHER * other_total)
    # ids: 63% of other rows on 3721-6519, 16% on 6520 (NEWS), 21% on 6521-13808
    total = sum(o["w"] for o in others) + news["w"]
    low, high = [], []
    acc = 0.0
    for o in others:
        (low if acc < 0.63 * total else high).append(o)
        acc += o["w"]
    for i, o in enumerate(low):
        o["id"] = int(3721 + i * (6519 - 3721) / max(len(low) - 1, 1))
    for i, o in enumerate(high):
        o["id"] = int(6521 + i * (13808 - 6521) / max(len(high) - 1, 1))
    return terms, others + [news]


def lab_values(rng, dist, n, female):
    kind = dist[0]
    lt = np.zeros(n, bool)
    gt = np.zeros(n, bool)
    if kind == "norm":
        _, mu, sd, lo, hi, dec = dist
        x = np.round(np.clip(rng.normal(mu, sd, n), lo, hi), dec)
    elif kind == "normsex":
        _, (mf, sf), (mm, sm), lo, hi, dec = dist
        x = np.where(female, rng.normal(mf, sf, n), rng.normal(mm, sm, n))
        x = np.round(np.clip(x, lo, hi), dec)
    elif kind == "lnorm":
        _, med, q95, lo, hi, dec = dist
        sig = np.log(q95 / med) / 1.645
        x = np.round(np.clip(med * np.exp(sig * rng.standard_normal(n)), lo, hi), dec)
    elif kind == "codes":
        _, vals, probs = dist
        x = np.array(vals, float)[choice_p(rng, n, probs)]
    elif kind == "sat":
        x = np.maximum(np.round(100 - rng.gamma(1.3, 2.2, n)), 70)
    elif kind == "crp":
        sig = np.log(200 / 15) / 1.645
        x = np.round(15 * np.exp(sig * rng.standard_normal(n)))
        lt = rng.random(n) < 0.45
        x[lt] = 3.0
        x = np.maximum(x, np.where(lt, 3.0, 4.0))
        gt = x >= 400
        x[gt] = 400
    elif kind == "trop":
        sig = np.log(250 / 14) / 1.645
        x = np.round(14 * np.exp(sig * rng.standard_normal(n)))
        x = np.maximum(x, 6)
        lt = rng.random(n) < 0.08
        x[lt] = 5.0
        gt = x >= 10000
        x[gt] = 10000
    else:
        raise ValueError(kind)
    return x, lt, gt


def make_lab(rng, P, V):
    n_persons = P["n"]
    terms, others = lab_catalogue()
    names = [t["name"] for t in terms] + [o["name"] for o in others]
    tid = vocab_index(names)
    cand = np.flatnonzero(np.isin(V["hosp"], [DS, SOS]) & ~V["err"] & (V["date"] >= LAB_START)
                          & (V["date"] <= LAB_END))
    nc = len(cand)
    a = (V["date"][cand] * 1440 + np.where(V["inexact"][cand], 12 * 60, V["minute"][cand])).astype(np.int64)
    T = np.nan_to_num(V["total"][cand], nan=191.0)
    adm = V["cls"][cand] == ADMIT
    end_ed = a + T

    # vital-sign rounds
    r_vis, r_t, r_first = [], [], []
    t = a + rng.integers(-60, 31, nc)
    r_vis.append(np.arange(nc))
    r_t.append(t)
    r_first.append(np.ones(nc, bool))
    cur = t.copy()
    active = np.ones(nc, bool)
    for _ in range(80):
        gap = np.where(cur < end_ed, rng.integers(60, 181, nc), rng.integers(240, 481, nc))
        nxt = cur + gap
        ok = active & ((nxt < end_ed) | (adm & (nxt < a + 2880)))
        if not ok.any():
            break
        idx = np.flatnonzero(ok)
        r_vis.append(idx)
        r_t.append(nxt[idx])
        r_first.append(np.zeros(len(idx), bool))
        cur = np.where(ok, nxt, cur)
        active = ok
    r_vis, r_t, r_first = np.concatenate(r_vis), np.concatenate(r_t), np.concatenate(r_first)
    on_hour = rng.random(len(r_t)) < 0.2
    r_t = np.where(on_hour, (r_t // 60) * 60, r_t)

    row_vis, row_term, row_min, row_sec = [], [], [], []
    for name, p in VITAL_ROUND_P.items():
        m = rng.random(len(r_vis)) < p
        row_vis.append(r_vis[m])
        row_term.append(np.full(m.sum(), tid[name]))
        row_min.append(r_t[m])
        row_sec.append(np.zeros(m.sum(), np.int64))
    row_vis.append(r_vis[r_first])
    row_term.append(np.full(r_first.sum(), tid["Medvetandegrad enl Retts"]))
    row_min.append(r_t[r_first])
    row_sec.append(np.zeros(r_first.sum(), np.int64))

    has_lab = rng.random(nc) < P_LAB_VISIT
    lab_t = a + rng.integers(20, 91, nc)
    lv = np.flatnonzero(has_lab)
    for name, p in LAB_PANEL_P.items():
        m = lv[rng.random(len(lv)) < p]
        row_vis.append(m)
        row_term.append(np.full(len(m), tid[name]))
        row_min.append(lab_t[m])
        row_sec.append(rng.integers(0, 60, len(m)))
    main_rows = sum(len(x) for x in row_vis)
    mean_other = OTHER_TERMS_SHARE / (1 - OTHER_TERMS_SHARE) * main_rows / max(len(lv), 1)
    k_other = np.maximum(rng.poisson(mean_other, len(lv)), 0)
    ov = np.repeat(lv, k_other)
    ow = np.array([o["w"] for o in others])
    oterm = len(terms) + choice_p(rng, len(ov), ow)
    obs = np.array([o["kind"] == "obs" for o in others])[oterm - len(terms)]
    # observations land on a random vital round of the visit, labs at sampling time
    first_round = np.searchsorted(r_vis[np.argsort(r_vis, kind="stable")], ov)
    sorted_rounds = np.argsort(r_vis, kind="stable")
    n_rounds = np.bincount(r_vis, minlength=nc)
    pick = first_round + np.floor(rng.random(len(ov)) * n_rounds[ov]).astype(np.int64)
    otime = np.where(obs, r_t[sorted_rounds[np.minimum(pick, len(r_t) - 1)]], lab_t[ov])
    row_vis.append(ov)
    row_term.append(oterm)
    row_min.append(otime)
    row_sec.append(np.where(obs, 0, rng.integers(0, 60, len(ov))))

    row_vis = np.concatenate(row_vis)
    row_term = np.concatenate(row_term)
    row_min = np.concatenate(row_min)
    row_sec = np.concatenate(row_sec)

    # choose visits so that rows hit 7.64 per person and follow the lab date quantiles
    rows_per_visit = np.bincount(row_vis, minlength=nc)
    target_rows = int(round(ROWS_PER_PERSON["lab"] * n_persons))
    # each month gets its share of the target rows; within a month Danderyd visits are favoured
    month = ((V["date"][cand] - LAB_START) // 30.44).astype(np.int64)
    nm = month.max() + 1
    edges = LAB_START + np.arange(nm + 1) * 30.44
    month_target = np.diff(cdf(edges, knots(LAB_DATE_KNOTS))) * target_rows
    w = np.where(V["hosp"][cand] == DS, LAB_DS_WEIGHT, 1.0)
    keys = np.log(rng.random(nc)) / w
    sel = np.zeros(nc, bool)
    carry = 0.0
    for m in range(nm):
        vis = np.flatnonzero(month == m)
        if len(vis) == 0:
            continue
        vis = vis[np.argsort(-keys[vis])]
        cum = np.cumsum(rows_per_visit[vis])
        take = np.searchsorted(cum, month_target[m] + carry) + 1
        sel[vis[:take]] = True
        carry += month_target[m] - cum[min(take, len(cum)) - 1]
    keep = sel[row_vis]
    row_vis, row_term, row_min, row_sec = row_vis[keep], row_term[keep], row_min[keep], row_sec[keep]
    n = len(row_vis)
    visit = cand[row_vis]
    person = V["person"][visit]

    # deliberate error: 1% of rows 2-30 days away from the visit
    disp = rng.random(n) < ERR_LAB_DISPLACED
    shift = (2 + rng.random(disp.sum()) * 28) * 1440
    sign = np.where(rng.random(disp.sum()) < 0.5, -1, 1)
    dm = row_min[disp]
    later = dm + shift
    death_min = P["death"][person[disp]] * 1440
    sign = np.where((later >= np.minimum(death_min, (LAB_END + 1) * 1440)), -1, sign)
    row_min[disp] = (dm + sign * shift).astype(np.int64)

    # nothing after death or outside the extract window
    day = row_min // 1440
    ok = (day <= np.minimum(P["death"][person], LAB_END)) & (row_min >= LAB_START * 1440 + LAB_MIN_MINUTE)
    row_vis, row_term, row_min, row_sec, visit, person = (x[ok] for x in (row_vis, row_term, row_min, row_sec,
                                                                          visit, person))
    n = len(row_vis)
    inexact = rng.random(n) < 0.0000294
    row_min = np.where(inexact, (row_min // 1440) * 1440, row_min)
    row_sec = np.where(inexact, 0, row_sec)
    us = (row_min * 60 + row_sec) * 1_000_000
    order = np.lexsort((us, person))
    row_vis, row_term, us, visit, person, inexact = (x[order] for x in (row_vis, row_term, us, visit, person, inexact))

    # source label by period and hospital
    q_ds2 = np.quantile(us, 0.0997)
    q_ds4 = np.quantile(us, 1 - 0.2919)
    period = np.where(us < q_ds2, 0, np.where(us >= q_ds4, 2, 1))
    is_ds = V["hosp"][visit] == DS
    from_vocab = ["ds2_danderyd", "ds2_södersjukhuset", "ds3_danderyd", "ds3_södersjukhuset", "ds4_danderyd",
                  "ds4_södersjukhuset"]
    src = period * 2 + np.where(is_ds, 0, 1)

    all_terms = terms + others
    female = ~P["male"][person]
    value = np.full(n, np.nan)
    lt = np.zeros(n, bool)
    gt = np.zeros(n, bool)
    unit = np.full(n, -1)
    ref = np.full(n, -1)
    unit_vocab, ref_vocab = [], []
    uidx, ridx = {}, {}
    tids = np.full(n, np.nan)
    for j, t in enumerate(all_terms):
        rows = np.flatnonzero(row_term == j)
        if len(rows) == 0:
            continue
        x, l, g = lab_values(rng, t["dist"], len(rows), female[rows])
        value[rows], lt[rows], gt[rows] = x, l, g
        tids[rows] = t["id"]
        if t["unit"]:
            unit[rows] = uidx.setdefault(t["unit"], len(uidx))
        rf, rm = t["ref"]
        if rf:
            a_ = ridx.setdefault(rf, len(ridx))
            b_ = ridx.setdefault(rm, len(ridx))
            ref[rows] = np.where(female[rows], a_, b_)
    unit_vocab = list(uidx)
    ref_vocab = list(ridx)

    # term_id: null for every ds4 row plus a share of the others
    ds4 = period == 2
    p_extra = (0.3166 - ds4.mean()) / max(1 - ds4.mean(), 1e-9)
    tids[ds4 | (rng.random(n) < p_extra)] = np.nan

    kinds = np.array([t["kind"] for t in all_terms])[row_term]
    labrow = np.flatnonzero(kinds == "lab")
    oflag = np.full(n, -1)
    n_flag = int(round((1 - 0.9978) * n))
    fr = rng.choice(labrow, min(n_flag, len(labrow)), replace=False)
    oflag[fr] = choice_p(rng, len(fr), OTHER_FLAG_W)
    vstr = np.full(n, -1)
    other_lab = np.flatnonzero((kinds == "lab") & (row_term >= len(terms)) & (oflag < 0))
    vs = rng.choice(other_lab, max(1, int(round(294 / REAL_ROWS["lab"] * n))), replace=False)
    vstr[vs] = choice_p(rng, len(vs), VALUE_STR_P)
    value[(oflag >= 0) | (vstr >= 0)] = np.nan
    lt[(oflag >= 0) | (vstr >= 0)] = False
    gt[(oflag >= 0) | (vstr >= 0)] = False
    value[rng.random(n) < 0.00004] = np.nan
    garbage = np.flatnonzero((rng.random(n) < LAB_GARBAGE_RATE) & ~np.isnan(value))
    if len(garbage) >= 2:
        g = rng.uniform(-140, 922000, len(garbage)).round()
        g[0], g[1] = -140, 922000
        value[garbage] = g

    cols = [str_col(src, from_vocab), pa.array(P["id"][person]), pa.array(inexact), ts_col(us),
            str_col(row_term, names), f64_col(tids), str_col(vstr, VALUE_STR), f64_col(value),
            str_col(unit, unit_vocab), pa.array(lt), pa.array(gt), str_col(oflag, OTHER_FLAGS), str_col(ref, ref_vocab)]
    return pa.Table.from_arrays(cols, schema=LAB_SCHEMA)


# =============================================================================
# Driver
# =============================================================================


def write(table, path, version, row_group):
    pq.write_table(table, path, compression="snappy", version=version, row_group_size=row_group)
    got = pq.read_schema(path)
    want = table.schema
    if not got.equals(want, check_metadata=False):
        raise SystemExit(f"schema mismatch in {path}:\n{got}\nvs\n{want}")
    log(f"wrote {path.name}: {table.num_rows:,} rows x {table.num_columns} columns")


def generate(args):
    rng = np.random.Generator(np.random.PCG64(args.seed))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    P = make_persons(rng, args.n_persons)
    log(f"persons: {P['n']:,}, deceased {P['die'].sum():,}")
    C = build_complaints(rng)
    V_DX_KEY[:] = C["dx"]
    V = make_visits(rng, P, C)
    log(f"ED visits: {V['n']:,}")
    pools = build_pools(rng)
    DP = death_pools(rng)
    DC = assign_death_causes(rng, P, DP)

    ed = make_ed(rng, P, V, C)
    write(ed, out / FILES["ed"], "1.0", max(ed.num_rows, 1))
    del ed
    lab = make_lab(rng, P, V)
    write(lab, out / FILES["lab"], "2.6", 1_000_000)
    del lab
    death = make_death(rng, P, V, DC, DP)
    write(death, out / FILES["death"], "1.0", max(death.num_rows, 1))
    del death
    patreg = make_patreg_tables(rng, P, V, DC, pools)
    write(patreg, out / FILES["patreg"], "1.0", max(patreg.num_rows, 1))


# =============================================================================
# Section 9: validation
# =============================================================================

def dates(*pairs):
    return {k: D(v) for k, v in pairs}


ED_TARGETS = dict(
    nulls={"ed_hospital": 0, "lopnr_swed": 0, "ed_arrival_datetime": 0, "ed_inexact_time": 0, "ed_age": 0,
           "is_male": 0, "ed_presented_complaint_std": 16.58, "ed_arrival_date": 0, "ed_arrival_time": 5.13,
           "ed_time_to_doctor_minutes": 40.94, "ed_time_total_minutes": 5.20, "ed_priority_in": 21.69,
           "ed_arrival_type": 36.60, "care_unit": 11.47, "ed_priority_out": 51.64, "ed_exit_to": 32.23,
           "ed_exit_to_care_unit": 90.35, "ed_presented_complaint_nonstd": 9.10},
    quantiles={
        "ed_arrival_date": dict(min="2003-04-02", q05="2005-10-20", q25="2009-07-14", q50="2012-09-07",
                                q75="2015-03-25", q95="2017-03-10", max="2017-12-18"),
        "ed_age": dict(min=18, q05=22, q25=35, q50=54, q75=72, q95=88, max=109),
        "ed_time_to_doctor_minutes": dict(q05=5, q25=26, q50=60, q75=125, q95=291, max=2880),
        "ed_time_total_minutes": dict(min=0, q05=33.08, q25=113, q50=191, q75=297, q95=527, max=2880),
        "ed_priority_in": dict(q05=1, q25=3, q50=3, q75=4, q95=5),
        "ed_priority_out": dict(q05=2, q25=3, q50=4, q75=4, q95=5),
    },
    categories={"ed_hospital": ("karolinska,1766422\nsödersjukhuset,1383033\nsahlgrenska,1260753\n"
                                "danderyd,1122655\nst_göran,931140", 12),
                "ed_presented_complaint_std": (LIST_A, 4924), "care_unit": (LIST_B, 90),
                "ed_arrival_type": (LIST_C, 46), "ed_exit_to": (LIST_D, 1004),
                "ed_exit_to_care_unit": (LIST_E, 452), "ed_presented_complaint_nonstd": (LIST_F, 10225),
                "ed_arrival_time": (None, 1440)},
    rates={"ed_inexact_time": 5.13, "is_male": 45.9},
    pmfs={"ed_priority_in": {0: .01, 1: .05, 2: .17, 3: .36, 4: .30, 5: .11},
          "ed_priority_out": {0: .005, 1: .015, 2: .08, 3: .39, 4: .33, 5: .18}},
)

PATREG_TARGETS = dict(
    nulls={**{"from": 0, "source_row": 0, "lopnr_swed": 0, "INDATUM": 0, "UTDATUM": 87.20, "Hdia": 14.50,
              "care_level": 0, "SJUKHUS": 0.04, "KTYP": 61.40, "INSATT": 87.25, "UTSATT": 87.23, "DIA_ANT": 57.60,
              "OP_ANT": 78.10, "is_female": 0, "SENINV": 79.01, "SENUTV": 96.02},
           **{f"DIAGNOS{i}": 100 - r for i, r in enumerate(DIAG_RATES, 1) if i > 1}, "DIAGNOS1": 14.45,
           **{f"OP{i}": 100 - r for i, r in enumerate(OP_RATES, 1)},
           **{f"EKOD{i}": 100 - r for i, r in enumerate(EKOD_RATES, 1)},
           "OPD1": 100 - 5.830, "OPD2": 100 - 3.341, "OPD3": 100 - 1.881, "OPD4": 100 - 1.086, "OPD5": 100 - 0.643},
    quantiles={
        "INDATUM": dict(min="1996-01-01", q05="2001-10-17", q25="2005-12-13", q50="2010-05-04", q75="2013-11-19",
                        q95="2016-05-12", max="2016-12-31"),
        "UTDATUM": dict(min="1996-10-15", q05="1998-09-09", q25="2004-06-08", q50="2009-04-09", q75="2013-03-15",
                        q95="2016-03-18", max="2016-12-31"),
        "KTYP": dict(q75=0, q95=8),
        "DIA_ANT": dict(q50=1, q75=2, q95=4, max=65),
        "OP_ANT": dict(q50=1, q75=2, q95=4, max=603),
    },
    categories={"Hdia": (LIST_G, 2411), "DIAGNOS1": (LIST_G, 2410), "DIAGNOS2": (LIST_H, None),
                "DIAGNOS4": (LIST_I, None), "EKOD1": (LIST_J, 12201), "OP1": (LIST_K, 9715),
                "OP2": (LIST_L, None), "SJUKHUS": (LIST_M, 1077), "SENINV": (None, 24893),
                "SENUTV": (None, 10438), "INSATT": (None, 3), "UTSATT": (None, 5), "care_level": (None, 2)},
    rates={"is_female": 57.9},
    pmfs={"KTYP": {0: .85, 1: .03, 2: .02, 3: .01, 4: .01, 5: .005, 6: .005, 7: .01, 8: .04, 9: .02},
          "INSATT": {"3": .871, "1": .121, "2": .008},
          "UTSATT": {"3": .846, "1": .114, "2": .026, "4": .014},
          "care_level": {"spec_outpatient": .872, "inpatient": .128}},
)

DEATH_TARGETS = dict(
    nulls={**{"from": 0, "source_row": 0, "lopnr_swed": 0, "inexact_death_date": 0, "date_death_meanimp": 0,
              "date_death_minimp": 0, "ULORSAK": 10.20, "KAP19": 94.94, "LK": 0.01, "DODSPL": 18.00,
              "DBGRUND1": 89.94, "is_female": 0},
           **{f"MORSAK{i}": 100 - r for i, r in enumerate(MORSAK_RATES, 1)}},
    quantiles={
        "date_death_meanimp": dict(min="2003-03-19", q05="2006-01-05", q25="2009-09-14", q50="2012-11-30",
                                   q75="2015-07-10", q95="2017-06-25", max="2017-12-31"),
        "date_death_minimp": dict(min="2003-01-01", q05="2006-01-05", q25="2009-09-14", q50="2012-11-30",
                                  q75="2015-07-10", q95="2017-06-25", max="2017-12-31"),
        "DODSPL": dict(q25=1, q50=2, q75=2, q95=3),
        "DBGRUND1": dict(q25=1, q50=3, q75=3),
    },
    categories={"ULORSAK": (LIST_N, 2177), "MORSAK1": (LIST_O, None), "MORSAK2": (LIST_P, None),
                "MORSAK4": (LIST_Q, None), "KAP19": (LIST_R, 472), "LK": (LIST_S, 290)},
    rates={"inexact_death_date": 0.80, "is_female": 52.1},
    pmfs={"DODSPL": {1: .45, 2: .42, 3: .11, 4: .02}, "DBGRUND1": {1: .47, 2: .02, 3: .49, 4: .02}},
)

LAB_TARGETS = dict(
    nulls={"from": 0, "lopnr_swed": 0, "inexact_time": 0, "datetime_analysis": 0, "term_name": 0, "term_id": 31.66,
           "value_str": 99.998, "value_num": 0.23, "unit": 71.29, "lt_flag": 0, "gt_flag": 0, "other_flag": 99.78,
           "ref_val1": 68.58},
    quantiles={
        "datetime_analysis": dict(min="2005-01-01", q05="2012-02-22", q25="2013-11-25", q50="2015-08-18",
                                  q75="2016-09-07", q95="2017-06-13", max="2017-08-31"),
        "term_id": dict(min=402, q05=402, q25=1995, q50=2025, q75=3720, q95=6520, max=13808),
        "value_num": dict(min=-140, q05=0.58, q25=10.5, q50=60, q75=99, q95=162, max=922000),
    },
    categories={"from": ("ds3_danderyd,5349999\nds3_södersjukhuset,4749133\nds4_södersjukhuset,2523132\n"
                         "ds4_danderyd,2322277", 6),
                "term_name": ("value,rows\n" + "\n".join(f"{k},{v}" for k, v in LIST_T_SHARES.items()), 150),
                "unit": (LIST_V, 20), "ref_val1": (LIST_W, 95),
                "other_flag": ("hemolys,13280\ninsuf_amount,9795\nclot,7406\nerror,2866\ncomment,2357\n"
                               "not_performed,1016", None)},
    rates={"inexact_time": 0.003, "lt_flag": 1.10, "gt_flag": 0.016},
    pmfs={},
)


class Report:
    def __init__(self):
        self.rows = []

    def add(self, check, target, observed, ok):
        self.rows.append((check, str(target), str(observed), "pass" if ok else "FAIL"))

    def print_and_save(self, path):
        w = max(len(r[0]) for r in self.rows)
        for r in self.rows:
            print(f"{r[3]:4s}  {r[0]:<{w}s}  target {r[1]:<28s} observed {r[2]}")
        with open(path, "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["check", "target", "observed", "result"])
            cw.writerows(self.rows)
        fails = sum(r[3] == "FAIL" for r in self.rows)
        print(f"\n{len(self.rows) - fails} passed, {fails} failed; report saved to {path}")
        return fails


def to_num(arr):
    t = arr.type
    if pa.types.is_date32(t):
        return arr.cast(pa.int32()).to_numpy(zero_copy_only=False).astype(float)
    if pa.types.is_timestamp(t):
        return arr.cast(pa.int64()).to_numpy(zero_copy_only=False).astype(float) / 86_400e6
    return arr.to_numpy(zero_copy_only=False).astype(float)


def check_file(rep, name, table, targets, schema, real_rows, n_persons):
    rep.add(f"{name}: schema", "exact", "match" if table.schema.equals(schema, check_metadata=False) else "differs",
            table.schema.equals(schema, check_metadata=False))
    rpp = table.num_rows / n_persons
    want = real_rows / REAL_PERSONS
    rep.add(f"{name}: rows per person", f"{want:.4f} ±3%", f"{rpp:.4f}", abs(rpp / want - 1) <= 0.03)
    n = table.num_rows
    for col, target in targets["nulls"].items():
        obs = table[col].null_count / n * 100
        rep.add(f"{name}.{col}: null %", f"{target:.2f} ±1", f"{obs:.2f}", abs(obs - target) <= 1)
    for col, q in targets["quantiles"].items():
        arr = table[col].drop_null() if hasattr(table[col], "drop_null") else pc.drop_null(table[col])
        x = to_num(arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr)
        is_date = pa.types.is_date32(table[col].type) or pa.types.is_timestamp(table[col].type)
        for k, v in q.items():
            tv = D(v) if isinstance(v, str) else v
            if k == "min":
                obs = x.min()
                if is_date and pa.types.is_timestamp(table[col].type):
                    tv = tv + 14 / 24
                ok = obs >= tv - 1e-9
            elif k == "max":
                obs = x.max()
                ok = obs <= tv + (1 if is_date else 1e-9)
            else:
                obs = np.quantile(x, int(k[1:]) / 100, method="lower")
                ok = abs(obs - tv) <= 120 if is_date else abs(obs - tv) <= 0.1 * abs(tv) + 1e-9
            show = (lambda d: iso(np.floor(d))) if is_date else (lambda d: f"{d:g}")
            rep.add(f"{name}.{col}: {k}", show(tv), show(obs), ok)
    for col, (lst, distinct) in targets["categories"].items():
        vc = pc.value_counts(table[col].combine_chunks())
        vals = vc.field("values").to_pylist()
        cnts = np.array(vc.field("counts").to_pylist(), float)
        obs_share = {v: c / n * 100 for v, c in zip(vals, cnts) if v is not None}
        if lst is not None:
            text = lst if lst.startswith("value") else "value,rows\n" + lst
            lv, lc, _ = parse_list(text)
            top = np.argsort(-lc)[:10]
            for i in top:
                real = lc[i] / real_rows * 100
                o = obs_share.get(lv[i], 0.0)
                rep.add(f"{name}.{col} = {lv[i]!r}: %", f"{real:.2f} ±2", f"{o:.2f}", abs(o - real) <= 2)
        if distinct:
            nd = len(obs_share)
            rep.add(f"{name}.{col}: distinct", f"<= {distinct}", nd, nd <= distinct)
    for col, target in targets["rates"].items():
        x = to_num(table[col].combine_chunks().cast(pa.float64()) if table[col].type == pa.bool_()
                   else table[col].combine_chunks())
        obs = np.nanmean(x) * 100
        rep.add(f"{name}.{col}: rate %", f"{target:.3f} ±1", f"{obs:.3f}", abs(obs - target) <= 1)
    for col, pmf in targets["pmfs"].items():
        arr = pc.drop_null(table[col].combine_chunks())
        vals = arr.to_pylist()
        tot = len(vals)
        for k, p in pmf.items():
            share = sum(1 for v in vals if v == k or (isinstance(v, float) and v == float(k))) / tot \
                if isinstance(k, str) else float(np.mean(np.asarray(vals, float) == k))
            if isinstance(k, str):
                share = float(np.mean(np.asarray(vals, dtype=object) == k))
            rep.add(f"{name}.{col} = {k}: share %", f"{p * 100:.2f} ±1", f"{share * 100:.2f}",
                    abs(share - p) <= 0.01)


def validate(args):
    out = Path(args.out_dir)
    rep = Report()
    rng = np.random.Generator(np.random.PCG64(args.seed))
    ids = np.sort(rng.choice(ID_MAX, size=args.n_persons, replace=False) + 1).astype(np.float64)
    n_persons = args.n_persons
    ed = pq.read_table(out / FILES["ed"])
    check_file(rep, "ed", ed, ED_TARGETS, ED_SCHEMA, REAL_ROWS["ed"], n_persons)
    death = pq.read_table(out / FILES["death"])
    check_file(rep, "death", death, DEATH_TARGETS, DEATH_SCHEMA, REAL_ROWS["death"], n_persons)
    lab = pq.read_table(out / FILES["lab"])
    check_file(rep, "lab", lab, LAB_TARGETS, LAB_SCHEMA, REAL_ROWS["lab"], n_persons)
    patreg = pq.read_table(out / FILES["patreg"])
    check_file(rep, "patreg", patreg, PATREG_TARGETS, PATREG_SCHEMA, REAL_ROWS["patreg"], n_persons)

    # --- IDs
    ed_id = ed["lopnr_swed"].to_numpy()
    ed_set = np.unique(ed_id)
    for name, t in (("ed", ed), ("patreg", patreg), ("death", death), ("lab", lab)):
        u = np.unique(t["lopnr_swed"].to_numpy())
        bad = (~np.isin(u, ids)).sum()
        rep.add(f"ids: {name} ids belong to generated persons", 0, bad, bad == 0)
    missing = (~np.isin(ids, ed_set)).sum()
    rep.add("ids: persons without an ED row", 0, missing, missing == 0)
    did = death["lopnr_swed"].to_numpy()
    rep.add("ids: death rows per deceased person", 1, f"{len(did) - len(np.unique(did))} duplicates",
            len(did) == len(np.unique(did)))

    # --- ED internals
    age = ed["ed_age"].to_numpy()
    rep.add("ed: min age", ">= 18", age.min(), age.min() >= 18)
    us = ed["ed_arrival_datetime"].combine_chunks().cast(pa.int64()).to_numpy()
    edate = ed["ed_arrival_date"].combine_chunks().cast(pa.int32()).to_numpy()
    same = (us // 86_400_000_000 == edate).mean()
    rep.add("ed: arrival_date = date(arrival_datetime)", "100%", f"{same * 100:.3f}%", same == 1)
    inex = ed["ed_inexact_time"].to_numpy(zero_copy_only=False)
    midnight = (us[inex] % 86_400_000_000 == 0).all()
    tnull = ed["ed_arrival_time"].is_null().to_numpy(zero_copy_only=False)
    rep.add("ed: inexact rows at 00:00:00", "all", "all" if midnight else "not all", bool(midnight))
    rep.add("ed: arrival_time null exactly when inexact", "exact", f"{(tnull == inex).mean() * 100:.3f}%",
            bool((tnull == inex).all()))
    doc = ed["ed_time_to_doctor_minutes"].to_numpy(zero_copy_only=False)
    tot = ed["ed_time_total_minutes"].to_numpy(zero_copy_only=False)
    both = ~np.isnan(doc) & ~np.isnan(tot)
    okshare = (doc[both] <= tot[both]).mean() * 100
    rep.add("ed: time_to_doctor <= time_total", "99.5% ±0.3", f"{okshare:.2f}%", abs(okshare - 99.5) <= 0.3)

    # --- register internals
    h = patreg["Hdia"].to_numpy(zero_copy_only=False)
    d1 = patreg["DIAGNOS1"].to_numpy(zero_copy_only=False)
    both = np.array([a is not None and b is not None for a, b in zip(h, d1)])
    eq = np.mean(h[both] == d1[both]) * 100
    rep.add("patreg: DIAGNOS1 = Hdia where both set", ">= 99.9%", f"{eq:.3f}%", eq >= 99.9)
    ind = patreg["INDATUM"].combine_chunks().cast(pa.int32()).to_numpy()
    utd_arr = patreg["UTDATUM"].combine_chunks()
    utd_null = utd_arr.is_null().to_numpy(zero_copy_only=False)
    utd = utd_arr.cast(pa.int32()).fill_null(0).to_numpy()
    rep.add("patreg: UTDATUM >= INDATUM", "all", int((utd[~utd_null] < ind[~utd_null]).sum()),
            bool((utd[~utd_null] >= ind[~utd_null]).all()))
    outpat = (patreg["care_level"].to_numpy(zero_copy_only=False) == "spec_outpatient")
    rep.add("patreg: UTDATUM null exactly on outpatient rows", "exact", int((utd_null != outpat).sum()),
            bool((utd_null == outpat).all()))
    for prefix, k in (("DIAGNOS", 30), ("OP", 30), ("EKOD", 7)):
        nul = np.column_stack([patreg[f"{prefix}{i}"].is_null().to_numpy(zero_copy_only=False)
                               for i in range(1, k + 1)])
        gaps = (nul[:, :-1] & ~nul[:, 1:]).any(axis=1).sum()
        rep.add(f"patreg: {prefix}n filled without gaps", 0, int(gaps), gaps == 0)
    nul = np.column_stack([death[f"MORSAK{i}"].is_null().to_numpy(zero_copy_only=False) for i in range(1, 49)])
    gaps = (nul[:, :-1] & ~nul[:, 1:]).any(axis=1).sum()
    rep.add("death: MORSAKn filled without gaps", 0, int(gaps), gaps == 0)

    # --- cross-file timing (exact death dates only)
    exact = ~death["inexact_death_date"].to_numpy(zero_copy_only=False)
    dd = dict(zip(did[exact], death["date_death_meanimp"].combine_chunks().cast(pa.int32()).to_numpy()[exact]))
    pid = patreg["lopnr_swed"].to_numpy()
    uts = patreg["UTSATT"].to_numpy(zero_copy_only=False)
    m4 = np.flatnonzero(uts == "4")
    ok4 = np.mean([dd.get(pid[i]) == utd[i] for i in m4]) * 100 if len(m4) else 100
    rep.add("cross: UTSATT '4' -> death date = UTDATUM", "100%", f"{ok4:.2f}% of {len(m4)}", ok4 == 100)
    cu = ed["ed_exit_to_care_unit"].to_numpy(zero_copy_only=False)
    mo = np.flatnonzero(cu == "OBDUK")
    oko = np.mean([dd.get(ed_id[i]) == edate[i] for i in mo]) * 100 if len(mo) else 100
    rep.add("cross: 'OBDUK' -> death date = arrival date", "100%", f"{oko:.2f}% of {len(mo)}", oko == 100)
    n_dead = len(dd)
    dead_ids = np.array(list(dd.keys()))
    dead_dates = np.array(list(dd.values()))

    def after_death(ids_, days_):
        o = np.argsort(dead_ids)
        pos = np.searchsorted(dead_ids[o], ids_)
        pos = np.clip(pos, 0, len(o) - 1)
        hit = dead_ids[o][pos] == ids_
        late = hit & (days_ > dead_dates[o][pos])
        return np.unique(ids_[late])

    ed_after = len(after_death(ed_id, edate)) / n_dead * 100
    rep.add("cross: deceased with an ED row after death", "~0.1% (<=0.3)", f"{ed_after:.3f}%", ed_after <= 0.3)
    pr_after = len(after_death(pid, ind)) / n_dead * 100
    rep.add("cross: deceased with an episode after death", "~0.05% (<=0.2)", f"{pr_after:.3f}%", pr_after <= 0.2)
    lid = lab["lopnr_swed"].to_numpy()
    lus = lab["datetime_analysis"].combine_chunks().cast(pa.int64()).to_numpy()
    lab_after = len(after_death(lid, lus // 86_400_000_000))
    rep.add("cross: deceased with lab rows after death", 0, lab_after, lab_after == 0)

    # --- lab linkage
    hosp = ed["ed_hospital"].to_numpy(zero_copy_only=False)
    ds = np.isin(hosp, ["danderyd", "södersjukhuset"])
    vis_id, vis_us = ed_id[ds], us[ds]
    linked = np.isin(np.unique(lid), np.unique(vis_id)).all()
    rep.add("lab: persons have a danderyd/SÖS ED visit", "all", "all" if linked else "not all", bool(linked))
    key_v = vis_id * 1e12 + vis_us / 1e6
    o = np.argsort(key_v)
    key_v = key_v[o]
    key_l = lid * 1e12 + lus / 1e6
    pos = np.searchsorted(key_v, key_l)
    best = np.full(len(key_l), np.inf)
    for cand in (pos - 1, pos):
        c = np.clip(cand, 0, len(key_v) - 1)
        same_p = np.floor(key_v[c] / 1e12) == lid
        dist = np.where(same_p, np.abs(key_l - key_v[c]), np.inf)
        best = np.minimum(best, dist)
    within = (best <= 48 * 3600 + 3600).mean() * 100
    rep.add("lab: rows within 48 h of a danderyd/SÖS visit", ">= 99% (target ~99)", f"{within:.2f}%",
            within >= 98.5)

    # --- outcome realism
    vkey = np.unique(np.column_stack([ed_id, edate]), axis=0)
    o = np.argsort(dead_ids)
    pos = np.clip(np.searchsorted(dead_ids[o], vkey[:, 0]), 0, len(o) - 1)
    hit = dead_ids[o][pos] == vkey[:, 0]
    gap = dead_dates[o][pos] - vkey[:, 1]
    followed = (hit & (gap >= 0) & (gap <= 30)).mean() * 100
    rep.add("outcome: ED visits followed by death within 30 days", "1.5 ±0.5 %", f"{followed:.2f}%",
            abs(followed - 1.5) <= 0.5)
    with_visit = len(np.unique(vkey[:, 0][hit & (gap >= 0) & (gap <= 30)])) / n_dead * 100
    rep.add("outcome: deceased with an ED visit in last 30 days", "40 ±5 %", f"{with_visit:.1f}%",
            abs(with_visit - 40) <= 5)

    return rep.print_and_save(out / "validation_report.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-persons", type=int, default=50_000)
    ap.add_argument("--out-dir", default="./synthetic")
    ap.add_argument("--validate", action="store_true", help="check the written files against the spec")
    args = ap.parse_args(argv)
    if args.validate:
        sys.exit(1 if validate(args) else 0)
    generate(args)


if __name__ == "__main__":
    main()