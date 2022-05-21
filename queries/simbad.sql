/*
    <WORK> (c) by Rik Ghosh, Soham Saha

    <WORK> is licensed under a
    Creative Commons Attribution 4.0 International License.

    You should have received a copy of the license along with this
    work. If not, see https://creativecommons.org/licenses/by/4.0/
*/

-- Collect Photometric Data from SIMBAD 4 database
-- link: http://simbad.cds.unistra.fr/simbad/sim-tap

SELECT
    B.RA,                           -- right ascension (J2000)
    B.DEC,                          -- declination (J2000)
    B.plx_value as parallax,        -- trigonometric parallax
    F.U,                            -- U band
    F.B,                            -- B band
    F.V,                            -- V band
    F.R,                            -- R band
    F.I                             -- I band

FROM basic AS B
JOIN allfluxes AS F ON F.oidref = B.oid
WHERE
CONTAINS(
    POINT('ICRS', B.RA, B.DEC),
    CIRCLE('ICRS', 345.67348, 59.55911, 0.16666666666666666)
)=1