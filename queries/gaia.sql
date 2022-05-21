/*
    <WORK> (c) by Rik Ghosh, Soham Saha

    <WORK> is licensed under a
    Creative Commons Attribution 4.0 International License.

    You should have received a copy of the license along with this
    work. If not, see https://creativecommons.org/licenses/by/4.0/
*/

-- Collect Astrometric and Photometric Data from GAIA Gaia Source Table from EDR3
-- link: https://gea.esac.esa.int/archive/

SELECT
    GS.ra,                              -- right ascension (J2000)
    GS.ra_error,                        -- right ascension error
    GS.dec,                             -- declination (J2000)
    GS.dec_error,                       -- declination error
    GS.parallax,                        -- parallax
    GS.parallax_error,                  -- parallax error
    GS.pm,                              -- proper motion
    GS.pmra,                            -- proper motion in right ascension
    GS.pmra_error,                      -- pmra error
    GS.pmdec,                           -- proper motion in declination
    GS.pmdec_error,                     -- pmdec error
    GS.phot_g_mean_mag AS g,            -- G band
    GS.phot_bp_mean_mag AS bp,          -- BP band
    GS.phot_rp_mean_mag AS rp,          -- RP band
    GS.bp_rp,                           -- BP - RP
    GS.bp_g,                            -- BP - G
    GS.g_rp                             -- G - RP

FROM gaiaedr3.gaia_source AS GS
WHERE 
CONTAINS(
	POINT('ICRS', GS.ra, GS.dec),
	CIRCLE('ICRS', 345.67348, 59.55911, 0.16666666666666666)
)=1