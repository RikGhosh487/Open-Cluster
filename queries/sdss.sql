/*
    <WORK> (c) by Rik Ghosh, Soham Saha

    <WORK> is licensed under a
    Creative Commons Attribution 4.0 International License.

    You should have received a copy of the license along with this
    work. If not, see https://creativecommons.org/licenses/by/4.0/
*/

-- Collect Photometric Data from the SDSS PhotoObj View from Data Release 17
-- link: http://skyserver.sdss.org/dr17/SearchTools/SQL/

SELECT
    P.ra,                       -- right ascension (J2000)
    P.raErr AS ra_err,          -- right ascension error
    P.dec,                      -- declination (J2000)
    P.decErr AS dec_err,        -- declination error
    P.psfMag_u AS u,            -- u band
    P.psfMagErr_u AS u_err,     -- u band error
    P.psfMag_g AS g,            -- g band
    P.psfMagErr_g AS g_err,     -- g band error
    P.psfMag_r AS r,            -- r band
    P.psfMagErr_r AS r_err,     -- r band error
    P.psfMag_i AS i,            -- i band
    P.psfMagErr_i AS i_err,     -- i band error
    P.psfMag_z AS z,            -- z band
    P.psfMagErr_g AS g_err      -- z band error

FROM PhotoObj AS P
JOIN dbo.fGetNearbyObjEq(345.67348, 59.55911, 10) AS CN
ON P.objID = CN.objID