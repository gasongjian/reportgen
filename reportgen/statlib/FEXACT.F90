MODULE Types

!      ALGORITHM 643, COLLECTED ALGORITHMS FROM ACM.
!      THIS WORK PUBLISHED IN TRANSACTIONS ON MATHEMATICAL SOFTWARE,
!      VOL. 19, NO. 4, DECEMBER, 1993, PP. 484-488.

! This version by amiller @ bigpond.net.au
! http://users.bigpond.net.au/amiller

! Latest revision - 1 March 2003

IMPLICIT NONE

INTEGER, PARAMETER, PUBLIC :: dp = SELECTED_REAL_KIND(14, 60)

END MODULE


MODULE Fisher_Exact

USE Types

CONTAINS


!-----------------------------------------------------------------------
!  Name:       FEXACT

!  Purpose:    Computes Fisher's exact test probabilities and a hybrid
!              approximation to Fisher exact test probabilities for a
!              contingency table using the network algorithm.

!  Usage:      CALL FEXACT (NROW, NCOL, TABLE, LDTABL, EXPECT, PERCNT,
!                          EMIN, WKSPACE, PRT, PRE)

!  Arguments:
!     NROW   - The number of rows in the table.  (Input)
!     NCOL   - The number of columns in the table.  (Input)
!     TABLE  - NROW by NCOL matrix containing the contingency table. (Input)
!     LDTABL - Leading dimension of TABLE exactly as specified in the
!              dimension statement in the calling program.  (Input)
!     EXPECT - Expected value used in the hybrid algorithm for
!              deciding when to use asymptotic theory probabilities. (Input)
!              If EXPECT <= 0.0 then asymptotic theory probabilities
!              are not used and Fisher exact test probabilities are
!              computed.  Otherwise, if PERCNT or more of the cells in
!              the remaining table have estimated expected values of
!              EXPECT or more, with no remaining cell having expected
!              value less than EMIN, then asymptotic chi-squared
!              probabilities are used.  See the algorithm section of the
!              manual document for details.  Use EXPECT = 5.0 to obtain
!              the 'Cochran' condition.
!     PERCNT - Percentage of remaining cells that must have estimated
!              expected  values greater than EXPECT before asymptotic
!              probabilities can be used.  (Input)
!              See argument EXPECT for details.  Use PERCNT = 80.0 to
!              obtain the 'Cochran' condition.
!     EMIN   - Minimum cell estimated expected value allowed for
!              asymptotic chi-squared probabilities to be used.  (Input)
!              See argument EXPECT for details.  Use EMIN = 1.0 to
!              obtain the 'Cochran' condition.
!     WKSPACE - Workspace size (Input)
!     PRT    - Probability of the observed table for fixed marginal
!              totals.  (Output)
!     PRE    - Table p-value.  (Output)
!              PRE is the probability of a more extreme table, where
!              'extreme' is in a probabilistic sense.
!              If EXPECT < 0 then the Fisher exact probability is returned.
!              Otherwise, an approximation to the Fisher exact probability is
!              computed based upon asymptotic chi-squared probabilities for
!              ``large'' table expected values.  The user defines ``large''
!              through the arguments EXPECT, PERCNT, and EMIN.

!  Remarks:
!  1. For many problems one megabyte or more of workspace can be required.
!     If the environment supports it, the user should begin by increasing
!     the workspace used to 200,000 units.

!  2. In FEXACT, LDSTP = 30*LDKEY.  The proportion of table space used by STP
!     may be changed by changing the line MULT = 30 below to another value.

!  3. FEXACT may be converted to single precision by setting IREAL = 3,
!     and converting all REAL (dp) specifications (except the
!     specifications for RWRK, IWRK, and DWRK) to REAL.  This will
!     require changing the names and specifications of the intrinsic
!     functions ALOG, AMAX1, AMIN1, EXP, and REAL.  In addition, the
!     machine specific constants will need to be changed, and the name
!     DWRK will need to be changed to RWRK in the call to F2XACT.

!  4. Machine specific constants are specified and documented in F2XACT.
!     A missing value code is specified in both FEXACT and F2XACT.

!  5. Although not a restriction, is is not generally practical to call
!     this routine with large tables which are not sparse and in which the
!     'hybrid' algorithm has little effect.  For example, although it is
!     feasible to compute exact probabilities for the table
!            1 8 5 4 4 2 2
!            5 3 3 4 3 1 0
!           10 1 4 0 0 0 0,
!     computing exact probabilities for a similar table which has been
!     enlarged by the addition of an extra row (or column) may not be feasible.
!-----------------------------------------------------------------------

SUBROUTINE fexact (nrow, ncol, table, ldtabl, expect, percnt, emin, wkspace, prt, pre)

!                                  SPECIFICATIONS FOR ARGUMENTS

INTEGER, INTENT(IN)     :: nrow
INTEGER, INTENT(IN)     :: ncol
REAL (dp), INTENT(IN)   :: table(:,:)
INTEGER, INTENT(IN)     :: ldtabl
REAL (dp), INTENT(IN)   :: expect
REAL (dp), INTENT(IN)   :: percnt
REAL (dp), INTENT(IN)   :: emin
INTEGER, OPTIONAL, INTENT(IN)     :: wkspace
REAL (dp), INTENT(OUT)  :: prt
REAL (dp), INTENT(OUT)  :: pre

!                                  SPECIFICATIONS FOR LOCAL VARIABLES
INTEGER   :: i, j, ldkey, ldstp, mult, nco, nro, ntot
REAL (dp) :: amiss
!                                  SPECIFICATIONS FOR INTRINSICS
! INTRINSIC  MAX0
! INTEGER :: MAX0
!                                  SPECIFICATIONS FOR SUBROUTINES
! EXTERNAL   prterr, f2xact
!                                  SPECIFICATIONS FOR FUNCTIONS
! EXTERNAL   iwork
! INTEGER :: iwork
!***********************************************************************
!                                  To increase the length of the table
!                                  of paste path lengths relative to the
!                                  length of the hash table, increase MULT
!***********************************************************************
mult   = 30
IF(PRESENT(wkspace)) THEN
  mult = wkspace
END IF
!***********************************************************************
!                                  AMISS is a missing value indicator which
!                                  is returned when the probability is not
!                                  defined.
!***********************************************************************
amiss = -12345.0D0

IF (nrow > ldtabl) THEN
  CALL prterr (1, 'NROW must be less than or equal to LDTABL.')
END IF
ntot = 0
DO i=1, nrow
  DO j=1, ncol
    IF (table(i,j) < 0) THEN
      CALL prterr (2, 'All elements of TABLE must be positive.')
    END IF
    ntot = ntot + table(i,j)
  END DO
END DO
IF (ntot == 0) THEN
  CALL prterr (3, 'All elements of TABLE are zero. '//  &
               'PRT and PRE are set to missing values (NaN, not a number).')
  prt = amiss
  pre = amiss
  GO TO 9000
END IF

nco = MAX(nrow,ncol)
nro = nrow + ncol - nco
ldkey = 500
ldstp = mult * ldkey

CALL f2xact (nrow, ncol, table, ldtabl, expect, percnt, emin,  &
             prt, pre, ldkey, ldstp, ntot, nco, nro)

9000 RETURN
END SUBROUTINE fexact


!-----------------------------------------------------------------------
!  Name:       F2XACT

!  Purpose:    Computes Fisher's exact test for a contingency table,
!              routine with workspace variables specified.

!  Usage:      CALL F2XACT (NROW, NCOL, TABLE, LDTABL, EXPECT, PERCNT, EMIN,
!                           PRT, PRE, LDKEY, LDSTP, NTOT, NCO, NRO)

!  N.B. Arguments FACT, ICO, IRO, KYY, IDIF, IRN, KEY, IPOIN, STP, IFRQ, DLP,
!       DSP, TM, KEY2, IWK & RWK have been removed, while arguments
!       LDKEY, LDSTP, NTOT, NCO & NRO have been added.
!-----------------------------------------------------------------------

SUBROUTINE f2xact (nrow, ncol, table, ldtabl, expect, percnt, emin, prt,  &
                   pre, ldkey, ldstp, ntotal, ncols, nrows)

!                                  SPECIFICATIONS FOR ARGUMENTS

INTEGER, INTENT(IN)     :: nrow
INTEGER, INTENT(IN)     :: ncol
REAL (dp), INTENT(IN)   :: table(:,:)
INTEGER, INTENT(IN)     :: ldtabl
REAL (dp), INTENT(IN)   :: expect
REAL (dp), INTENT(IN)   :: percnt
REAL (dp), INTENT(IN)   :: emin
REAL (dp), INTENT(OUT)  :: prt, pre
INTEGER, INTENT(IN)     :: ldkey, ldstp, ntotal, ncols, nrows

!                                  SPECIFICATIONS FOR LOCAL VARIABLES

INTEGER   :: ico(ncols), iro(ncols), kyy(ncols), idif(nrows), irn(nrows),  &
             key(2*ldkey), ipoin(2*ldkey), ifrq(6*ldstp), key2(2*ldkey)
REAL (dp) :: fact(0:ntotal), stp(2*ldstp), dlp(2*ldkey), dsp(2*ldkey),  &
             tm(2*ldkey)

INTEGER   :: i, iflag, ifreq, ii, ikkey, ikstp, ikstp2, ipn, ipo, itmp,  &
             itop, itp, j, jkey, jstp, jstp2, jstp3, jstp4, k, k1, kb, kd, &
             kmax, ks, kval, last, n, ncell, nco, nrb, nro, nro2, ntot
REAL (dp) :: dd, ddf, df, drn, dro, dspt, emn, obs, obs2, obs3, pastp, pv, tmp
LOGICAL   :: chisq, ipsh

!                                  SPECIFICATIONS FOR INTRINSICS
! INTRINSIC  DLOG, DMAX1, DMIN1, DEXP, MAX0, MIN0, MOD, NINT, DBLE
! INTEGER :: MAX, MIN0, MOD, nint
! REAL (dp) :: DLOG, DMAX1, DMIN1, DEXP, DBLE

!                                  SPECIFICATIONS FOR SUBROUTINES
! EXTERNAL   prterr, f3xact, f4xact, f5xact, f6xact, f7xact, isort

!                                  SPECIFICATIONS FOR FUNCTIONS
! EXTERNAL   f9xact, gammds
! REAL (dp) :: f9xact, gammds
!***********************************************************************
!                                  IMAX is the largest representable
!                                  integer on the machine
!***********************************************************************
! DATA imax/2147483647/
INTEGER, PARAMETER :: imax = HUGE(1)
!***********************************************************************
!                                  AMISS is a missing value indicator
!                                  which is returned when the
!                                  probability is not defined.
!***********************************************************************
REAL (dp), PARAMETER :: amiss = -12345.0_dp
!***********************************************************************
!                                  TOL is chosen as the square root of
!                                  the smallest relative spacing
!***********************************************************************
REAL (dp), PARAMETER :: tol = 3.45254e-07_dp
!***********************************************************************
!                                  EMX is a large positive value used
!                                  in comparing expected values
!***********************************************************************
REAL (dp), PARAMETER :: emx = 1.0e+30_dp
!                                  Initialize KEY array
DO i=1, 2*ldkey
  key(i)  = -9999
  key2(i) = -9999
END DO
!                                  Initialize parameters
pre  = 0.0
IF (expect > 0.0D0) THEN
  emn = emin
ELSE
  emn = emx
END IF
!                                  Check table dimensions
IF (nrow > ldtabl) THEN
  CALL prterr (1, 'NROW must be less than or equal to LDTABL.')
END IF
IF (ncol <= 1) THEN
  CALL prterr (4, 'NCOL must be greater than 1.0.')
END IF
!                                  Compute row marginals and total
ntot = 0
DO i=1, nrow
  iro(i) = 0
  DO j=1, ncol
    IF (table(i,j) < -0.0001D0) THEN
      CALL prterr (2, 'All elements of TABLE must be positive.')
    END IF
    iro(i) = iro(i) + nint(table(i,j))
    ntot   = ntot + nint(table(i,j))
  END DO
END DO

IF (ntot == 0) THEN
  CALL prterr (3, 'All elements of TABLE are zero. '//  &
               'PRT and PRE are set to missing values (NaN, not a number).')
  prt = amiss
  pre = amiss
  GO TO 9000
END IF
!                                  Column marginals
DO i=1, ncol
  ico(i) = 0
  DO j=1, nrow
    ico(i) = ico(i) + nint(table(j,i))
  END DO
END DO
!                                  sort
CALL isort (nrow, iro)
CALL isort (ncol, ico)
!                                  Determine row and column marginals

IF (nrow > ncol) THEN
  nro = ncol
  nco = nrow
!                                  Interchange row and column marginals
  DO i=1, nrow
    itmp = iro(i)
    IF (i <= ncol) iro(i) = ico(i)
    ico(i) = itmp
  END DO
ELSE
  nro = nrow
  nco = ncol
END IF

!                                  Get multiplers for stack
kyy(1) = 1
DO i=2, nro
!                                  Hash table multipliers
  IF (iro(i-1)+1 <= imax/kyy(i-1)) THEN
    kyy(i) = kyy(i-1)*(iro(i-1)+1)
    j      = j/kyy(i-1)
  ELSE
    CALL prterr (5, 'The hash table key cannot be computed because the'//   &
                 ' largest key is larger than the largest representable'//  &
                 ' integer.  The algorithm cannot proceed.')
  END IF
END DO
!                                  Maximum product
IF (iro(nro-1)+1 <= imax/kyy(nro-1)) THEN
  kmax = (iro(nro)+1)*kyy(nro-1)
ELSE
  CALL prterr (5, 'The hash table key cannot be computed because the'//   &
               ' largest key is larger than the largest representable'//  &
               ' integer.  The algorithm cannot proceed.')
  GO TO 9000
END IF
!                                  Compute log factorials
fact(0) = 0.0D0
fact(1) = 0.0D0
fact(2) = LOG(2.0D0)
DO i=3, ntot, 2
  fact(i) = fact(i-1) + LOG(DBLE(i))
  j       = i + 1
  IF (j <= ntot) fact(j) = fact(i) + fact(2) + fact(j/2) - fact(j/2-1)
END DO
!                                  Compute observed path length: OBS
obs  = tol
ntot = 0
DO j=1, nco
  dd = 0.0
  DO i=1, nro
    IF (nrow <= ncol) THEN
      dd   = dd + fact(nint(table(i,j)))
      ntot = ntot + nint(table(i,j))
    ELSE
      dd   = dd + fact(nint(table(j,i)))
      ntot = ntot + nint(table(j,i))
    END IF
  END DO
  obs = obs + fact(ico(j)) - dd
END DO
!                                  Denominator of observed table: DRO
dro = f9xact(nro, ntot, iro, fact)
prt = EXP(obs-dro)
!                                  Initialize pointers
k        = nco
last     = ldkey + 1
jkey     = ldkey + 1
jstp     = ldstp + 1
jstp2    = 3*ldstp + 1
jstp3    = 4*ldstp + 1
jstp4    = 5*ldstp + 1
ikkey    = 0
ikstp    = 0
ikstp2   = 2*ldstp
ipo      = 1
ipoin(1) = 1
stp(1)   = 0.0
ifrq(1)  = 1
ifrq(ikstp2+1) = -1

110 kb = nco - k + 1
ks   = 0
n    = ico(kb)
kd   = nro + 1
kmax = nro
!                                  IDIF is the difference in going to the
!                                  daughter
idif(1:nro) = 0
!                                  Generate the first daughter
130 kd = kd - 1
ntot     = MIN(n, iro(kd))
idif(kd) = ntot
IF (idif(kmax) == 0) kmax = kmax - 1
n = n - ntot
IF (n > 0 .AND. kd /= 1) GO TO 130
IF (n /= 0) GO TO 310

k1   = k - 1
n    = ico(kb)
ntot = 0
DO i=kb + 1, nco
  ntot = ntot + ico(i)
END DO
!                                  Arc to daughter length = ICO(KB)
150 DO i=1, nro
  irn(i) = iro(i) - idif(i)
END DO
!                                  Sort irn
IF (k1 > 1) THEN
  IF (nro == 2) THEN
    IF (irn(1) > irn(2)) THEN
      ii     = irn(1)
      irn(1) = irn(2)
      irn(2) = ii
    END IF
  ELSE IF (nro == 3) THEN
    ii = irn(1)
    IF (ii > irn(3)) THEN
      IF (ii > irn(2)) THEN
        IF (irn(2) > irn(3)) THEN
          irn(1) = irn(3)
          irn(3) = ii
        ELSE
          irn(1) = irn(2)
          irn(2) = irn(3)
          irn(3) = ii
        END IF
      ELSE
        irn(1) = irn(3)
        irn(3) = irn(2)
        irn(2) = ii
      END IF
    ELSE IF (ii > irn(2)) THEN
      irn(1) = irn(2)
      irn(2) = ii
    ELSE IF (irn(2) > irn(3)) THEN
      ii     = irn(2)
      irn(2) = irn(3)
      irn(3) = ii
    END IF
  ELSE
    DO j=2, nro
      i  = j - 1
      ii = irn(j)
      170 IF (ii < irn(i)) THEN
        irn(i+1) = irn(i)
        i        = i - 1
        IF (i > 0) GO TO 170
      END IF
      irn(i+1) = ii
    END DO
  END IF
!                                  Adjust start for zero
  DO i=1, nro
    IF (irn(i) /= 0) EXIT
  END DO
  nrb = i
  nro2 = nro - i + 1
ELSE
  nrb  = 1
  nro2 = nro
END IF
!                                  Some table values
ddf = f9xact(nro, n, idif, fact)
drn = f9xact(nro2, ntot, irn(nrb:), fact) - dro + ddf
!                                  Get hash value
IF (k1 > 1) THEN
  kval = irn(1) + irn(2)*kyy(2)
  DO i=3, nro
    kval = kval + irn(i)*kyy(i)
  END DO
!                                  Get hash table entry
  i = MOD(kval, 2*ldkey) + 1
!                                  Search for unused location
  DO itp=i, 2*ldkey
    ii = key2(itp)
    IF (ii == kval) THEN
      GO TO 240
    ELSE IF (ii < 0) THEN
      key2(itp) = kval
      dlp(itp)  = 1.0D0
      dsp(itp)  = 1.0D0
      GO TO 240
    END IF
  END DO

  DO itp=1, i - 1
    ii = key2(itp)
    IF (ii == kval) THEN
      GO TO 240
    ELSE IF (ii < 0) THEN
      key2(itp) = kval
      dlp(itp)  = 1.0
      GO TO 240
    END IF
  END DO

  CALL prterr (6, 'LDKEY is too small.  It is not possible to '//  &
               'give thevalue of LDKEY required, but you could '//  &
               'try doubling LDKEY (and possibly LDSTP).')
END IF

240 ipsh = .true.
!                                  Recover pastp
ipn   = ipoin(ipo+ikkey)
pastp = stp(ipn+ikstp)
ifreq = ifrq(ipn+ikstp)
!                                  Compute shortest and longest path
IF (k1 > 1) THEN
  obs2 = obs - fact(ico(kb+1)) - fact(ico(kb+2)) - ddf
  DO i=3, k1
    obs2 = obs2 - fact(ico(kb+i))
  END DO

  IF (dlp(itp) > 0.0D0) THEN
    dspt = obs - obs2 - ddf
!                                  Compute longest path
    dlp(itp) = 0.0D0
    CALL f3xact (nro2, irn(nrb:), k1, ico(kb+1:), dlp(itp), ntot, fact, tol)
    dlp(itp) = MIN(0.0D0,dlp(itp))
!                                  Compute shortest path
    dsp(itp) = dspt
    CALL f4xact (nro2, irn(nrb:), k1, ico(kb+1:), dsp(itp), fact, tol)
    dsp(itp) = MIN(0.0_dp, dsp(itp)-dspt)
!                                  Use chi-squared approximation?
    IF (DBLE(irn(nrb)*ico(kb+1))/DBLE(ntot) > emn) THEN
      ncell = 0.0
      DO i=1, nro2
        DO j=1, k1
          IF (irn(nrb+i-1)*ico(kb+j) >= ntot*expect) THEN
            ncell = ncell + 1
          END IF
        END DO
      END DO
      IF (ncell*100 >= k1*nro2*percnt) THEN
        tmp = 0.0
        DO i=1, nro2
          tmp = tmp + fact(irn(nrb+i-1)) - fact(irn(nrb+i-1)-1)
        END DO
        tmp = tmp*(k1-1)
        DO j=1, k1
          tmp = tmp + (nro2-1)*(fact(ico(kb+j)) - fact(ico(kb+j)-1))
        END DO
        df      = (nro2-1)*(k1-1)
        tmp     = tmp + df*1.83787706640934548356065947281D0
        tmp     = tmp - (nro2*k1-1)*(fact(ntot) - fact(ntot-1))
        tm(itp) = -2.0D0*(obs-dro) - tmp
      ELSE
!                                  tm(itp) set to a flag value
        tm(itp) = -9876.0D0
      END IF
    ELSE
      tm(itp) = -9876.0D0
    END IF
  END IF
  obs3 = obs2 - dlp(itp)
  obs2 = obs2 - dsp(itp)
  IF (tm(itp) == -9876.0D0) THEN
    chisq = .false.
  ELSE
    chisq = .true.
    tmp   = tm(itp)
  END IF
ELSE
  obs2 = obs - drn - dro
  obs3 = obs2
END IF
!                                  Process node with new PASTP
300 IF (pastp <= obs3) THEN
!                                  Update pre
  pre = pre + DBLE(ifreq)*EXP(pastp+drn)

ELSE IF (pastp < obs2) THEN
  IF (chisq) THEN
    df  = (nro2-1)*(k1-1)
    pv  = 1.0 - gammad( MAX(0.0D0, tmp+2.0D0*(pastp+drn))/2.0D0, df/2.0D0)
    pre = pre + DBLE(ifreq)*EXP(pastp+drn)*pv
  ELSE
!                                  Put daughter on queue
    CALL f5xact (pastp+ddf, tol, kval, key(jkey:), ldkey, ipoin(jkey:),  &
                 stp(jstp:), ldstp, ifrq(jstp:), ifrq(jstp2:),   &
                 ifrq(jstp3:), ifrq(jstp4:), ifreq, itop, ipsh)
    ipsh = .false.
  END IF
END IF
!                                  Get next PASTP on chain
ipn = ifrq(ipn+ikstp2)
IF (ipn > 0) THEN
  pastp = stp(ipn+ikstp)
  ifreq = ifrq(ipn+ikstp)
  GO TO 300
END IF
!                                  Generate a new daughter node
CALL f7xact (kmax, iro, idif, kd, ks, iflag)
IF (iflag /= 1) GO TO 150
!                                  Go get a new mother from stage K
310 iflag = 1
CALL f6xact (nro, iro, iflag, kyy, key(ikkey+1:), ldkey, last, ipo)
!                                  Update pointers
IF (iflag == 3) THEN
  k      = k - 1
  itop   = 0
  ikkey  = jkey - 1
  ikstp  = jstp - 1
  ikstp2 = jstp2 - 1
  jkey   = ldkey - jkey + 2
  jstp   = ldstp - jstp + 2
  jstp2  = 2*ldstp + jstp
  DO i=1, 2*ldkey
    key2(i) = -9999
  END DO
  IF (k >= 2) GO TO 310
ELSE
  GO TO 110
END IF

9000 RETURN
END SUBROUTINE f2xact


!-----------------------------------------------------------------------
!  Name:       F3XACT

!  Purpose:    Computes the shortest path length for a given table.

!  Usage:      CALL F3XACT (NROW, IROW, NCOL, ICOL, DLP, MM, FACT, TOL)

!  Arguments:
!     NROW   - The number of rows in the table.  (Input)
!     IROW   - Vector of length NROW containing the row sums for the
!              table.  (Input)
!     NCOL   - The number of columns in the table.  (Input)
!     ICOL   - Vector of length K containing the column sums for the
!              table.  (Input)
!     DLP    - The longest path for the table.  (Output)
!     MM     - The total count in the table.  (Output)
!     FACT   - Vector containing the logarithms of factorials.  (Input)
!     ICO    - Work vector of length MAX(NROW,NCOL).
!     IRO    - Work vector of length MAX(NROW,NCOL).
!     IT     - Work vector of length MAX(NROW,NCOL).
!     LB     - Work vector of length MAX(NROW,NCOL).
!     NR     - Work vector of length MAX(NROW,NCOL).
!     NT     - Work vector of length MAX(NROW,NCOL).
!     NU     - Work vector of length MAX(NROW,NCOL).
!     ITC    - Work vector of length 400.
!     IST    - Work vector of length 400.
!     STV    - Work vector of length 400.
!     ALEN   - Work vector of length MAX(NROW,NCOL).
!     TOL    - Tolerance.  (Input)

!  N.B. Arguments ICO, IRO, IT, LB, NR, NT, NU, ITC, IST, STV & ALEN
!       have been removed.
!-----------------------------------------------------------------------

SUBROUTINE f3xact (nrow, irow, ncol, icol, dlp, mm, fact, tol)

!                                  SPECIFICATIONS FOR ARGUMENTS

INTEGER, INTENT(IN)     :: nrow
INTEGER, INTENT(IN)     :: irow(:)
INTEGER, INTENT(IN)     :: ncol
INTEGER, INTENT(IN)     :: icol(:)
REAL (dp), INTENT(OUT)  :: dlp
INTEGER, INTENT(IN)     :: mm
REAL (dp), INTENT(IN)   :: fact(0:)
REAL (dp), INTENT(IN)   :: tol

!                                  SPECIFICATIONS FOR LOCAL VARIABLES

INTEGER, ALLOCATABLE   :: ico(:), iro(:), it(:), lb(:), nr(:), nt(:), nu(:)
REAL (dp), ALLOCATABLE :: alen(:)

INTEGER   :: i, ic1, ic2, ii, ipn, irl, ist(400), itc(400), itp, k, key, ks, &
             kyy, lev, n11, n12, nc1, nc1s, nco, nct, nn, nn1, nr1, nro, nrt
REAL (dp) :: stv(400), v, val, vmn
LOGICAL   :: xmin

!                                  SPECIFICATIONS FOR SAVE VARIABLES
INTEGER, SAVE :: ldst = 200, nitc = 0, nst = 0

!                                  SPECIFICATIONS FOR INTRINSICS
! INTRINSIC  DMIN1, INT, MOD, DBLE
! INTEGER :: INT, MOD
! REAL (dp) :: DMIN1, DBLE

!                                  SPECIFICATIONS FOR SUBROUTINES
! EXTERNAL   prterr, f10act, isort

nn = MAX(nrow, ncol)
ALLOCATE( ico(nn), iro(nn), it(nn), lb(nn), nr(nn), nt(nn), nu(nn),   &
          alen(0:nn) )

alen(0:ncol) = 0.0_dp
ist(1:400) = -1
!                                  nrow is 1
IF (nrow <= 1) THEN
  IF (nrow > 0) THEN
    dlp = dlp - fact(icol(1))
    DO i=2, ncol
      dlp = dlp - fact(icol(i))
    END DO
  END IF
  GO TO 9000
END IF
!                                  ncol is 1
IF (ncol <= 1) THEN
  IF (ncol > 0) THEN
    dlp = dlp - fact(irow(1)) - fact(irow(2))
    DO i=3, nrow
      dlp = dlp - fact(irow(i))
    END DO
  END IF
  GO TO 9000
END IF
!                                  2 by 2 table
IF (nrow*ncol == 4) THEN
  n11 = (irow(1)+1)*(icol(1)+1)/(mm+2)
  n12 = irow(1) - n11
  dlp = dlp - fact(n11) - fact(n12) - fact(icol(1)-n11) - fact(icol(2)-n12)
  GO TO 9000
END IF
!                                  Test for optimal table
val  = 0.0
xmin = .false.
IF (irow(nrow) <= irow(1)+ncol) THEN
  CALL f10act (nrow, irow, ncol, icol, val, xmin, fact, lb, nu, nr)
END IF
IF (.NOT.xmin) THEN
  IF (icol(ncol) <= icol(1)+nrow) THEN
    CALL f10act (ncol, icol, nrow, irow, val, xmin, fact, lb, nu, nr)
  END IF
END IF

IF (xmin) THEN
  dlp = dlp - val
  GO TO 9000
END IF
!                                  Setup for dynamic programming
nn = mm
!                                  Minimize ncol
IF (nrow >= ncol) THEN
  nro = nrow
  nco = ncol

  DO i=1, nrow
    iro(i) = irow(i)
  END DO

  ico(1) = icol(1)
  nt(1)  = nn - ico(1)
  DO i=2, ncol
    ico(i) = icol(i)
    nt(i)  = nt(i-1) - ico(i)
  END DO
ELSE
  nro = ncol
  nco = nrow

  ico(1) = irow(1)
  nt(1)  = nn - ico(1)
  DO i=2, nrow
    ico(i) = irow(i)
    nt(i)  = nt(i-1) - ico(i)
  END DO

  DO i=1, ncol
    iro(i) = icol(i)
  END DO
END IF
!                                  Initialize pointers
vmn  = 1.0D10
nc1s = nco - 1
irl  = 1
ks   = 0
k    = ldst
kyy  = ico(nco) + 1
GO TO 100
!                                  Test for optimality
90 xmin = .false.
IF (iro(nro) <= iro(irl)+nco) THEN
  CALL f10act (nro, iro(irl:), nco, ico, val, xmin, fact, lb, nu, nr)
END IF
IF (.NOT.xmin) THEN
  IF (ico(nco) <= ico(1)+nro) THEN
    CALL f10act (nco, ico, nro, iro(irl:), val, xmin, fact, lb, nu, nr)
  END IF
END IF

IF (xmin) THEN
  IF (val < vmn) vmn = val
  GO TO 200
END IF
!                                  Setup to generate new node
100 lev = 1
nr1   = nro - 1
nrt   = iro(irl)
nct   = ico(1)
lb(1) = INT(DBLE((nrt+1)*(nct+1))/DBLE(nn+nr1*nc1s+1)-tol) - 1
nu(1) = INT(DBLE((nrt+nc1s)*(nct+nr1))/DBLE(nn+nr1+nc1s)) - lb(1) + 1
nr(1) = nrt - lb(1)
!                                  Generate a node
110 nu(lev) = nu(lev) - 1
IF (nu(lev) == 0) THEN
  IF (lev == 1) GO TO 200
  lev = lev - 1
  GO TO 110
END IF
lb(lev) = lb(lev) + 1
nr(lev) = nr(lev) - 1
120 alen(lev) = alen(lev-1) + fact(lb(lev))
IF (lev < nc1s) THEN
  nn1     = nt(lev)
  nrt     = nr(lev)
  lev     = lev + 1
  nc1     = nco - lev
  nct     = ico(lev)
  lb(lev) = DBLE((nrt+1)*(nct+1))/DBLE(nn1+nr1*nc1+1) - tol
  nu(lev) = DBLE((nrt+nc1)*(nct+nr1))/DBLE(nn1+nr1+nc1) - lb(lev) + 1
  nr(lev) = nrt - lb(lev)
  GO TO 120
END IF
alen(nco) = alen(lev) + fact(nr(lev))
lb(nco)   = nr(lev)

v = val + alen(nco)
IF (nro == 2) THEN
!                                  Only 1 row left
  v = v + fact(ico(1)-lb(1)) + fact(ico(2)-lb(2))
  DO i=3, nco
    v = v + fact(ico(i)-lb(i))
  END DO
  IF (v < vmn) vmn = v
ELSE IF (nro == 3 .AND. nco == 2) THEN
!                                  3 rows and 2 columns
  nn1 = nn - iro(irl) + 2
  ic1 = ico(1) - lb(1)
  ic2 = ico(2) - lb(2)
  n11 = (iro(irl+1)+1)*(ic1+1)/nn1
  n12 = iro(irl+1) - n11
  v   = v + fact(n11) + fact(n12) + fact(ic1-n11) + fact(ic2-n12)
  IF (v < vmn) vmn = v
ELSE
!                                  Column marginals are new node
  DO i=1, nco
    it(i) = ico(i) - lb(i)
  END DO
!                                  Sort column marginals
  IF (nco == 2) THEN
    IF (it(1) > it(2)) THEN
      ii    = it(1)
      it(1) = it(2)
      it(2) = ii
    END IF
  ELSE IF (nco == 3) THEN
    ii = it(1)
    IF (ii > it(3)) THEN
      IF (ii > it(2)) THEN
        IF (it(2) > it(3)) THEN
          it(1) = it(3)
          it(3) = ii
        ELSE
          it(1) = it(2)
          it(2) = it(3)
          it(3) = ii
        END IF
      ELSE
        it(1) = it(3)
        it(3) = it(2)
        it(2) = ii
      END IF
    ELSE IF (ii > it(2)) THEN
      it(1) = it(2)
      it(2) = ii
    ELSE IF (it(2) > it(3)) THEN
      ii    = it(2)
      it(2) = it(3)
      it(3) = ii
    END IF
  ELSE
    CALL isort (nco, it)
  END IF
!                                  Compute hash value
  key = it(1)*kyy + it(2)
  DO i=3, nco
    key = it(i) + key*kyy
  END DO
!                                  Table index
  ipn = MOD(key,ldst) + 1
!                                  Find empty position
  ii = ks + ipn
  DO itp=ipn, ldst
    IF (ist(ii) < 0) THEN
      GO TO 180
    ELSE IF (ist(ii) == key) THEN
      GO TO 190
    END IF
    ii = ii + 1
  END DO

  ii = ks + 1
  DO itp=1, ipn - 1
    IF (ist(ii) < 0) THEN
      GO TO 180
    ELSE IF (ist(ii) == key) THEN
      GO TO 190
    END IF
    ii = ii + 1
  END DO

  CALL prterr (30, 'Stack length exceeded in f3xact.'//  &
               '  This problem should not occur.')
!                                  Push onto stack
  180 ist(ii) = key
  stv(ii) = v
  nst     = nst + 1
  ii      = nst + ks
  itc(ii) = itp
  GO TO 110
!                                  Marginals already on stack
  190 stv(ii) = MIN(v,stv(ii))
END IF
GO TO 110
!                                  Pop item from stack
200 IF (nitc > 0) THEN
!                                  Stack index
  itp      = itc(nitc+k) + k
  nitc     = nitc - 1
  val      = stv(itp)
  key      = ist(itp)
  ist(itp) = -1
!                                  Compute marginals
  DO i=nco, 2, -1
    ico(i) = MOD(key,kyy)
    key    = key/kyy
  END DO
  ico(1) = key
!                                  Set up nt array
  nt(1) = nn - ico(1)
  DO i=2, nco
    nt(i) = nt(i-1) - ico(i)
  END DO
  GO TO 90

ELSE IF (nro > 2 .AND. nst > 0) THEN
!                                  Go to next level
  nitc = nst
  nst  = 0
  k    = ks
  ks   = ldst - ks
  nn   = nn - iro(irl)
  irl  = irl + 1
  nro  = nro - 1
  GO TO 200
END IF

dlp = dlp - vmn

9000 DEALLOCATE( ico, iro, it, lb, nr, nt, nu, alen )
RETURN
END SUBROUTINE f3xact


!-----------------------------------------------------------------------
!  Name:       F4XACT

!  Purpose:    Computes the longest path length for a given table.

!  Usage:      CALL F4XACT (NROW, IROW, NCOL, ICOL, DSP, FACT, TOL)

!  Arguments:
!     NROW   - The number of rows in the table.  (Input)
!     IROW   - Vector of length NROW containing the row sums for the
!              table.  (Input)
!     NCOL   - The number of columns in the table.  (Input)
!     ICOL   - Vector of length K containing the column sums for the
!              table.  (Input)
!     DSP    - The shortest path for the table.  (Output)
!     FACT   - Vector containing the logarithms of factorials.  (Input)
!     ICSTK  - NCOL by NROW+NCOL+1 work array.
!     NCSTK  - Work vector of length NROW+NCOL+1.
!     LSTK   - Work vector of length NROW+NCOL+1.
!     MSTK   - Work vector of length NROW+NCOL+1.
!     NSTK   - Work vector of length NROW+NCOL+1.
!     NRSTK  - Work vector of length NROW+NCOL+1.
!     IRSTK  - NROW by MAX(NROW,NCOL) work array.
!     YSTK   - Work vector of length NROW+NCOL+1.
!     TOL    - Tolerance.  (Input)

!  N.B. Arguments ICSTK, NCSTK, LSTK, MSTK, NSTK, NRSTK, IRSTK & YSTK
!       hev been removed.
!-----------------------------------------------------------------------

SUBROUTINE f4xact (nrow, irow, ncol, icol, dsp, fact, tol)

!                                  SPECIFICATIONS FOR ARGUMENTS

INTEGER, INTENT(IN)        :: nrow
INTEGER, INTENT(IN)        :: irow(:)
INTEGER, INTENT(IN)        :: ncol
INTEGER, INTENT(IN)        :: icol(:)
REAL (dp), INTENT(IN OUT)  :: dsp
REAL (dp), INTENT(IN)      :: fact(0:)
REAL (dp), INTENT(IN)      :: tol

!                                  SPECIFICATIONS FOR LOCAL VARIABLES
INTEGER   :: icstk(ncol,nrow+ncol+1), ncstk(nrow+ncol+1), lstk(nrow+ncol+1), &
             mstk(nrow+ncol+1), nstk(nrow+ncol+1), nrstk(nrow+ncol+1),  &
             irstk(nrow,nrow+ncol+1)
REAL (dp) :: ystk(nrow+ncol+1)
INTEGER   :: i, ic1, ict, ir1, irt, istk, j, k, l, m, mn, n, nco, nro
REAL (dp) :: amx, y
!                                  SPECIFICATIONS FOR SUBROUTINES
! EXTERNAL   f11act, f8xact
!                                  Take care of the easy cases first
IF (nrow == 1) THEN
  DO i=1, ncol
    dsp = dsp - fact(icol(i))
  END DO
  GO TO 9000
END IF

IF (ncol == 1) THEN
  DO i=1, nrow
    dsp = dsp - fact(irow(i))
  END DO
  GO TO 9000
END IF

IF (nrow*ncol == 4) THEN
  IF (irow(2) <= icol(2)) THEN
    dsp = dsp - fact(irow(2)) - fact(icol(1)) - fact(icol(2)-irow(2))
  ELSE
    dsp = dsp - fact(icol(2)) - fact(irow(1)) - fact(irow(2)-icol(2))
  END IF
  GO TO 9000
END IF
!                                  initialization before loop
DO i=1, nrow
  irstk(i,1) = irow(nrow-i+1)
END DO

DO j=1, ncol
  icstk(j,1) = icol(ncol-j+1)
END DO

nro      = nrow
nco      = ncol
nrstk(1) = nro
ncstk(1) = nco
ystk(1)  = 0.0
y        = 0.0
istk     = 1
l        = 1
amx      = 0.0

50 ir1 = irstk(1,istk)
ic1 = icstk(1,istk)
IF (ir1 > ic1) THEN
  IF (nro >= nco) THEN
    m = nco - 1
    n = 2
  ELSE
    m = nro
    n = 1
  END IF
ELSE IF (ir1 < ic1) THEN
  IF (nro <= nco) THEN
    m = nro - 1
    n = 1
  ELSE
    m = nco
    n = 2
  END IF
ELSE
  IF (nro <= nco) THEN
    m = nro - 1
    n = 1
  ELSE
    m = nco - 1
    n = 2
  END IF
END IF

60 IF (n == 1) THEN
  i = l
  j = 1
ELSE
  i = 1
  j = l
END IF

irt = irstk(i,istk)
ict = icstk(j,istk)
mn  = irt
IF (mn > ict) mn = ict
y = y + fact(mn)
IF (irt == ict) THEN
  nro = nro - 1
  nco = nco - 1
  CALL f11act (irstk(:,istk), i, nro, irstk(:,istk+1))
  CALL f11act (icstk(:,istk), j, nco, icstk(:,istk+1))
ELSE IF (irt > ict) THEN
  nco = nco - 1
  CALL f11act (icstk(:,istk), j, nco, icstk(:,istk+1))
  CALL f8xact (irstk(:,istk), irt-ict, i, nro, irstk(:,istk+1))
ELSE
  nro = nro - 1
  CALL f11act (irstk(:,istk), i, nro, irstk(:,istk+1))
  CALL f8xact (icstk(:,istk), ict-irt, j, nco, icstk(:,istk+1))
END IF

IF (nro == 1) THEN
  DO k=1, nco
    y = y + fact(icstk(k,istk+1))
  END DO
  GO TO 90
END IF

IF (nco == 1) THEN
  DO k=1, nro
    y = y + fact(irstk(k,istk+1))
  END DO
  GO TO 90
END IF

lstk(istk)  = l
mstk(istk)  = m
nstk(istk)  = n
istk        = istk + 1
nrstk(istk) = nro
ncstk(istk) = nco
ystk(istk)  = y
l           = 1
GO TO 50

90 IF (y > amx) THEN
  amx = y
  IF (dsp-amx <= tol) THEN
    dsp = 0.0
    GO TO 9000
  END IF
END IF

100 istk = istk - 1
IF (istk == 0) THEN
  dsp = dsp - amx
  IF (dsp-amx <= tol) dsp = 0.0
  GO TO 9000
END IF
l = lstk(istk) + 1

110 IF (l > mstk(istk)) GO TO 100
n   = nstk(istk)
nro = nrstk(istk)
nco = ncstk(istk)
y   = ystk(istk)
IF (n == 1) THEN
  IF (irstk(l,istk) < irstk(l-1,istk)) GO TO 60
ELSE IF (n == 2) THEN
  IF (icstk(l,istk) < icstk(l-1,istk)) GO TO 60
END IF

l = l + 1
GO TO 110

9000 RETURN
END SUBROUTINE f4xact


!-----------------------------------------------------------------------
!  Name:       F5XACT

!  Purpose:    Put node on stack in network algorithm.

!  Usage:      CALL F5XACT (PASTP, TOL, KVAL, KEY, LDKEY, IPOIN, STP,
!                          LDSTP, IFRQ, NPOIN, NR, NL, IFREQ, ITOP,
!                          IPSH)

!  Arguments:
!     PASTP  - The past path length.  (Input)
!     TOL    - Tolerance for equivalence of past path lengths.  (Input)
!     KVAL   - Key value.  (Input)
!     KEY    - Vector of length LDKEY containing the key values. (Input/output)
!     LDKEY  - Length of vector KEY.  (Input)
!     IPOIN  - Vector of length LDKEY pointing to the linked list
!              of past path lengths.  (Input/output)
!     STP    - Vector of length LSDTP containing the linked lists
!              of past path lengths.  (Input/output)
!     LDSTP  - Length of vector STP.  (Input)
!     IFRQ   - Vector of length LDSTP containing the past path
!              frequencies.  (Input/output)
!     NPOIN  - Vector of length LDSTP containing the pointers to
!              the next past path length.  (Input/output)
!     NR     - Vector of length LDSTP containing the right object
!              pointers in the tree of past path lengths. (Input/output)
!     NL     - Vector of length LDSTP containing the left object
!              pointers in the tree of past path lengths. (Input/output)
!     IFREQ  - Frequency of the current path length.  (Input)
!     ITOP   - Pointer to the top of STP.  (Input)
!     IPSH   - Option parameter.  (Input)
!              If IPSH is true, the past path length is found in the table KEY.
!              Otherwise the location of the past path length is assumed
!              known and to have been found in a previous call.
!-----------------------------------------------------------------------

SUBROUTINE f5xact (pastp, tol, kval, key, ldkey, ipoin, stp,  &
                   ldstp, ifrq, npoin, nr, nl, ifreq, itop, ipsh)

!                                  SPECIFICATIONS FOR ARGUMENTS

REAL (dp), INTENT(IN)   :: pastp
REAL (dp), INTENT(IN)   :: tol
INTEGER, INTENT(IN)     :: kval
INTEGER, INTENT(OUT)    :: key(:)
INTEGER, INTENT(IN)     :: ldkey
INTEGER, INTENT(OUT)    :: ipoin(:)
REAL (dp), INTENT(OUT)  :: stp(:)
INTEGER, INTENT(IN)     :: ldstp
INTEGER, INTENT(OUT)    :: ifrq(:)
INTEGER, INTENT(OUT)    :: npoin(:)
INTEGER, INTENT(OUT)    :: nr(:)
INTEGER, INTENT(OUT)    :: nl(:)
INTEGER, INTENT(IN)     :: ifreq
INTEGER, INTENT(IN OUT) :: itop
LOGICAL, INTENT(IN)     :: ipsh

!                                  SPECIFICATIONS FOR LOCAL VARIABLES
INTEGER   :: ipn, ird, itmp
REAL (dp) :: test1, test2
!                                  SPECIFICATIONS FOR SAVE VARIABLES
INTEGER, SAVE :: itp
!                                  SPECIFICATIONS FOR INTRINSICS
! INTRINSIC  MOD
! INTEGER :: MOD
!                                  SPECIFICATIONS FOR SUBROUTINES
! EXTERNAL   prterr

IF (ipsh) THEN
!                                  Convert KVAL to integer in range
!                                  1, ..., LDKEY.
  ird = MOD(kval,ldkey) + 1
!                                  Search for an unused location
  DO itp=ird, ldkey
    IF (key(itp) == kval) GO TO 40
    IF (key(itp) < 0) GO TO 30
  END DO

  DO itp=1, ird - 1
    IF (key(itp) == kval) GO TO 40
    IF (key(itp) < 0) GO TO 30
  END DO
!                                  Return if KEY array is full
  CALL prterr(6, 'LDKEY is too small for this problem.  It is '//  &
              'not possible to estimate the value of LDKEY '//  &
              'required, but twice the current value may be sufficient.')
!                                  Update KEY
  30 key(itp) = kval
  itop       = itop + 1
  ipoin(itp) = itop
!                                  Return if STP array full
  IF (itop > ldstp) THEN
    CALL prterr(7, 'LDSTP is too small for this problem.  It is not '//  &
                'possible to estimate the value of LDSTP required,'//  &
                'but twice the current value may be sufficient.')
  END IF
!                                  Update STP, etc.
  npoin(itop) = -1
  nr(itop)    = -1
  nl(itop)    = -1
  stp(itop)   = pastp
  ifrq(itop)  = ifreq
  GO TO 9000
END IF
!                                  Find location, if any, of pastp
40 ipn = ipoin(itp)
test1 = pastp - tol
test2 = pastp + tol

50 IF (stp(ipn) < test1) THEN
  ipn = nl(ipn)
  IF (ipn > 0) GO TO 50
ELSE IF (stp(ipn) > test2) THEN
  ipn = nr(ipn)
  IF (ipn > 0) GO TO 50
ELSE
  ifrq(ipn) = ifrq(ipn) + ifreq
  GO TO 9000
END IF
!                                  Return if STP array full
itop = itop + 1
IF (itop > ldstp) THEN
  CALL prterr(7, 'LDSTP is too small for this problem.  It is '//  &
              'not possible to estimate the value of LDSTP '//  &
              'required, but twice the current value may be sufficient.')
  GO TO 9000
END IF
!                                  Find location to add value
ipn  = ipoin(itp)
itmp = ipn
60 IF (stp(ipn) < test1) THEN
  itmp = ipn
  ipn  = nl(ipn)
  IF (ipn > 0) THEN
    GO TO 60
  ELSE
    nl(itmp) = itop
  END IF
ELSE IF (stp(ipn) > test2) THEN
  itmp = ipn
  ipn  = nr(ipn)
  IF (ipn > 0) THEN
    GO TO 60
  ELSE
    nr(itmp) = itop
  END IF
END IF
!                                  Update STP, etc.
npoin(itop) = npoin(itmp)
npoin(itmp) = itop
stp(itop)   = pastp
ifrq(itop)  = ifreq
nl(itop)    = -1
nr(itop)    = -1

9000 RETURN
END SUBROUTINE f5xact


!-----------------------------------------------------------------------
!  Name:       F6XACT

!  Purpose:    Pop a node off the stack.

!  Usage:      CALL F6XACT (NROW, IROW, IFLAG, KYY, KEY, LDKEY, LAST, IPN)

!  Arguments:
!     NROW   - The number of rows in the table.  (Input)
!     IROW   - Vector of length nrow containing the row sums on output. (Output)
!     IFLAG  - Set to 3 if there are no additional nodes to process. (Output)
!     KYY    - Constant mutlipliers used in forming the hash table key. (Input)
!     KEY    - Vector of length LDKEY containing the hash table keys.
!              (Input/output)
!     LDKEY  - Length of vector KEY.  (Input)
!     LAST   - Index of the last key popped off the stack. (Input/output)
!     IPN    - Pointer to the linked list of past path lengths. (Output)
!-----------------------------------------------------------------------

SUBROUTINE f6xact (nrow, irow, iflag, kyy, key, ldkey, last, ipn)
!                                  SPECIFICATIONS FOR ARGUMENTS

INTEGER, INTENT(IN)       :: nrow
INTEGER, INTENT(OUT)      :: irow(:)
INTEGER, INTENT(OUT)      :: iflag
INTEGER, INTENT(IN)       :: kyy(:)
INTEGER, INTENT(IN OUT)   :: key(:)
INTEGER, INTENT(IN)       :: ldkey
INTEGER, INTENT(IN OUT)   :: last
INTEGER, INTENT(OUT)      :: ipn

!                                  SPECIFICATIONS FOR LOCAL VARIABLES
INTEGER :: j, kval
!                                  SPECIFICATIONS FOR SAVE VARIABLES

10 last = last + 1
IF (last <= ldkey) THEN
  IF (key(last) < 0) GO TO 10
!                                  Get KVAL from the stack
  kval      = key(last)
  key(last) = -9999
  DO j=nrow, 2, -1
    irow(j) = kval/kyy(j)
    kval    = kval - irow(j)*kyy(j)
  END DO
  irow(1) = kval
  ipn     = last
ELSE
  last  = 0
  iflag = 3
END IF
RETURN
END SUBROUTINE f6xact


!-----------------------------------------------------------------------
!  Name:       F7XACT

!  Purpose:    Generate the new nodes for given marinal totals.

!  Usage:      CALL F7XACT (NROW, IMAX, IDIF, K, KS, IFLAG)

!  Arguments:
!     NROW   - The number of rows in the table.  (Input)
!     IMAX   - The row marginal totals.  (Input)
!     IDIF   - The column counts for the new column.  (Input/output)
!     K      - Indicator for the row to decrement.  (Input/output)
!     KS     - Indicator for the row to increment.  (Input/output)
!     IFLAG  - Status indicator.  (Output)
!              If IFLAG is zero, a new table was generated.  For
!              IFLAG = 1, no additional tables could be generated.
!-----------------------------------------------------------------------

SUBROUTINE f7xact (nrow, imax, idif, k, ks, iflag)
!                                  SPECIFICATIONS FOR ARGUMENTS

INTEGER, INTENT(IN)     :: nrow
INTEGER, INTENT(IN)     :: imax(:)
INTEGER, INTENT(IN OUT) :: idif(:)
INTEGER, INTENT(IN OUT) :: k
INTEGER, INTENT(IN OUT) :: ks
INTEGER, INTENT(OUT)    :: iflag

!                                  SPECIFICATIONS FOR LOCAL VARIABLES
INTEGER :: i, k1, m, mm
!                                  SPECIFICATIONS FOR INTRINSICS
! INTRINSIC  MIN0
! INTEGER :: MIN0

iflag = 0
!                                  Find node which can be incremented, ks
IF (ks == 0) THEN
  10 ks = ks + 1
  IF (idif(ks) == imax(ks)) GO TO 10
END IF
!                                 Find node to decrement (>ks)
IF (idif(k) > 0 .AND. k > ks) THEN
  idif(k) = idif(k) - 1
  30 k = k - 1
  IF (imax(k) == 0) GO TO 30
  m = k
!                                 Find node to increment (>=ks)
  40 IF (idif(m) >= imax(m)) THEN
    m = m - 1
    GO TO 40
  END IF
  idif(m) = idif(m) + 1
!                                 Change ks
  IF (m == ks) THEN
    IF (idif(m) == imax(m)) ks = k
  END IF
ELSE
!                                 Check for finish
  50 DO k1=k + 1, nrow
    IF (idif(k1) > 0) GO TO 70
  END DO
  iflag = 1
  GO TO 9000
!                                 Reallocate counts
  70 mm = 1
  DO i=1, k
    mm      = mm + idif(i)
    idif(i) = 0
  END DO
  k = k1
  90 k = k - 1
  m       = MIN(mm,imax(k))
  idif(k) = m
  mm      = mm - m
  IF (mm > 0 .AND. k /= 1) GO TO 90
!                                 Check that all counts reallocated
  IF (mm > 0) THEN
    IF (k1 /= nrow) THEN
      k = k1
      GO TO 50
    END IF
    iflag = 1
    GO TO 9000
  END IF
!                                 Get ks
  idif(k1) = idif(k1) - 1
  ks       = 0
  100    ks = ks + 1
  IF (ks > k) GO TO 9000
  IF (idif(ks) >= imax(ks)) GO TO 100
END IF

9000 RETURN
END SUBROUTINE f7xact


!-----------------------------------------------------------------------
!  Name:       F8XACT

!  Purpose:    Routine for reducing a vector when there is a zero element.

!  Usage:      CALL F8XACT (IROW, IS, I1, IZERO, NEW)

!  Arguments:
!     IROW   - Vector containing the row counts.  (Input)
!     IS     - Indicator.  (Input)
!     I1     - Indicator.  (Input)
!     IZERO  - Position of the zero.  (Input)
!     NEW    - Vector of new row counts.  (Output)
!-----------------------------------------------------------------------

SUBROUTINE f8xact (irow, is, i1, izero, NEW)

!                                  SPECIFICATIONS FOR ARGUMENTS

INTEGER, INTENT(IN)   :: irow(:)
INTEGER, INTENT(IN)   :: is
INTEGER, INTENT(IN)   :: i1
INTEGER, INTENT(IN)   :: izero
INTEGER, INTENT(OUT)  :: NEW(:)

!                                  SPECIFICATIONS FOR LOCAL VARIABLES
INTEGER :: i

DO i=1, i1 - 1
  NEW(i) = irow(i)
END DO

DO i=i1, izero - 1
  IF (is >= irow(i+1)) GO TO 30
  NEW(i) = irow(i+1)
END DO

i = izero
30 NEW(i) = is
40 i = i + 1
IF (i > izero) RETURN
NEW(i) = irow(i)
GO TO 40
END SUBROUTINE f8xact


!-----------------------------------------------------------------------
!  Name:       F9XACT

!  Purpose:    Computes the log of a multinomial coefficient.

!  Usage:      F9XACT(N, MM, IR, FACT)

!  Arguments:
!     N      - Length of IR.  (Input)
!     MM     - Number for factorial in numerator.  (Input)
!     IR     - Vector of length N containing the numebers for the
!              denominator of the factorial.  (Input)
!     FACT   - Table of log factorials.  (Input)
!     F9XACT  - The log of the multinomal coefficient.  (Output)
!-----------------------------------------------------------------------

FUNCTION f9xact (n, mm, ir, fact) RESULT(fn_val)
!                                  SPECIFICATIONS FOR ARGUMENTS

INTEGER, INTENT(IN)    :: n
INTEGER, INTENT(IN)    :: mm
INTEGER, INTENT(IN)    :: ir(:)
REAL (dp), INTENT(IN)  :: fact(0:)
REAL (dp)              :: fn_val

!                                  SPECIFICATIONS FOR LOCAL VARIABLES
INTEGER :: k

fn_val = fact(mm)
DO k=1, n
  fn_val = fn_val - fact(ir(k))
END DO

RETURN
END FUNCTION f9xact


!-----------------------------------------------------------------------
!  Name:       F10ACT

!  Purpose:    Computes the shortest path length for special tables.

!  Usage:      CALL F10ACT (NROW, IROW, NCOL, ICOL, VAL, XMIN, FACT, ND,
!                          NE, M)

!  Arguments:
!     NROW   - The number of rows in the table.  (Input)
!     IROW   - Vector of length NROW containing the row totals.  (Input)
!     NCOL   - The number of columns in the table.  (Input)
!     ICO    - Vector of length NCOL containing the column totals. (Input)
!     VAL    - The shortest path.  (Output)
!     XMIN   - Set to true if shortest path obtained.  (Output)
!     FACT   - Vector containing the logarithms of factorials. (Input)
!     ND     - Workspace vector of length NROW.
!     NE     - Workspace vector of length NCOL.
!     M      - Workspace vector of length NCOL.

!  Chapter:    STAT/LIBRARY Categorical and Discrete Data Analysis
!-----------------------------------------------------------------------

SUBROUTINE f10act (nrow, irow, ncol, icol, val, xmin, fact, nd, NE, m)

!                                  SPECIFICATIONS FOR ARGUMENTS

INTEGER, INTENT(IN)     :: nrow
INTEGER, INTENT(IN)     :: irow(:)
INTEGER, INTENT(IN)     :: ncol
INTEGER, INTENT(IN)     :: icol(:)
REAL (dp), INTENT(OUT)  :: val
LOGICAL, INTENT(OUT)    :: xmin
REAL (dp), INTENT(IN)   :: fact(0:)
INTEGER, INTENT(OUT)    :: nd(:)
INTEGER, INTENT(OUT)    :: NE(:)
INTEGER, INTENT(OUT)    :: m(:)

!                                  SPECIFICATIONS FOR LOCAL VARIABLES
INTEGER :: i, is, ix, nrw1

DO i=1, nrow - 1
  nd(i) = 0
END DO

is    = icol(1)/nrow
NE(1) = is
ix    = icol(1) - nrow*is
m(1)  = ix
IF (ix /= 0) nd(ix) = nd(ix) + 1

DO i=2, ncol
  ix    = icol(i)/nrow
  NE(i) = ix
  is    = is + ix
  ix    = icol(i) - nrow*ix
  m(i)  = ix
  IF (ix /= 0) nd(ix) = nd(ix) + 1
END DO

DO i=nrow - 2, 1, -1
  nd(i) = nd(i) + nd(i+1)
END DO

ix   = 0
nrw1 = nrow + 1
DO i=nrow, 2, -1
  ix = ix + is + nd(nrw1-i) - irow(i)
  IF (ix < 0) RETURN
END DO

DO i=1, ncol
  ix  = NE(i)
  is  = m(i)
  val = val + is*fact(ix+1) + (nrow-is)*fact(ix)
END DO
xmin = .true.

RETURN
END SUBROUTINE f10act


!-----------------------------------------------------------------------
!  Name:       F11ACT

!  Purpose:    Routine for revising row totals.

!  Usage:      CALL F11ACT (IROW, I1, I2, NEW)

!  Arguments:
!     IROW   - Vector containing the row totals.  (Input)
!     I1     - Indicator.  (Input)
!     I2     - Indicator.  (Input)
!     NEW    - Vector containing the row totals.  (Input) ??
!-----------------------------------------------------------------------

SUBROUTINE f11act (irow, i1, i2, NEW)
!                                  SPECIFICATIONS FOR ARGUMENTS

INTEGER, INTENT(IN)   :: irow(:)
INTEGER, INTENT(IN)   :: i1
INTEGER, INTENT(IN)   :: i2
INTEGER, INTENT(OUT)  :: NEW(:)

!                                  SPECIFICATIONS FOR LOCAL VARIABLES
INTEGER :: i

DO i=1, i1 - 1
  NEW(i) = irow(i)
END DO

DO i=i1, i2
  NEW(i) = irow(i+1)
END DO

RETURN
END SUBROUTINE f11act


!-----------------------------------------------------------------------
!  Name:       ISORT

!  Purpose:    Shell sort for an integer vector.

!  Usage:      CALL ISORT (N, IX)

!  Arguments:
!     N      - Lenth of vector IX.  (Input)
!     IX     - Vector to be sorted.  (Input/output)
!-----------------------------------------------------------------------

SUBROUTINE isort (n, ix)
!                                  SPECIFICATIONS FOR ARGUMENTS

INTEGER, INTENT(IN)      :: n
INTEGER, INTENT(IN OUT)  :: ix(:)

!                                  SPECIFICATIONS FOR LOCAL VARIABLES
INTEGER :: i, ikey, il(10), it, iu(10), j, kl, ku, m

!                                  SPECIFICATIONS FOR SUBROUTINES
! EXTERNAL   prterr
!                                  Sort IX
m = 1
i = 1
j = n
10 IF (i >= j) GO TO 40
kl   = i
ku   = j
ikey = i
j    = j + 1
!                                  Find element in first half
20 i = i + 1
IF (i < j) THEN
  IF (ix(ikey) > ix(i)) GO TO 20
END IF
!                                  Find element in second half
30 j = j - 1
IF (ix(j) > ix(ikey)) GO TO 30
!                                  Interchange
IF (i < j) THEN
  it    = ix(i)
  ix(i) = ix(j)
  ix(j) = it
  GO TO 20
END IF
it       = ix(ikey)
ix(ikey) = ix(j)
ix(j)    = it
!                                  Save upper and lower subscripts of
!                                  the array yet to be sorted
IF (m < 11) THEN
  IF (j-kl < ku-j) THEN
    il(m) = j + 1
    iu(m) = ku
    i     = kl
    j     = j - 1
  ELSE
    il(m) = kl
    iu(m) = j - 1
    i     = j + 1
    j     = ku
  END IF
  m = m + 1
  GO TO 10
ELSE
  CALL prterr (20, 'This should never occur.')
END IF
!                                  Use another segment
40 m = m - 1
IF (m == 0) GO TO 9000
i = il(m)
j = iu(m)
GO TO 10

9000 RETURN
END SUBROUTINE isort


FUNCTION gammad(x, p) RESULT(fn_val)

!      ALGORITHM AS239  APPL. STATIST. (1988) VOL. 37, NO. 3

!      Computation of the Incomplete Gamma Integral

!      Auxiliary functions required: LNGAMMA = logarithm of the gamma
!      function, and ALNORM = algorithm AS66

! ELF90-compatible version by Alan Miller
! Latest revision - 29 November 1997

! N.B. Argument IFAULT has been removed

REAL (dp), INTENT(IN) :: x, p
REAL (dp)             :: fn_val

! Local variables
REAL (dp)             :: pn1, pn2, pn3, pn4, pn5, pn6, arg, c, rn, a, b, an
REAL (dp), PARAMETER  :: zero = 0._dp, one = 1._dp, two = 2._dp, &
                         oflo = 1.d+37, three = 3._dp, nine = 9._dp, &
                         tol = 1.d-14, xbig = 1.d+8, plimit = 1000._dp, &
                         elimit = -88._dp

fn_val = zero

!      Check that we have valid values for X and P

IF (p <= zero .OR. x < zero) THEN
  WRITE(*, *) 'AS239: Either p <= 0 or x < 0'
  RETURN
END IF
IF (x == zero) RETURN

!      Use a normal approximation if P > PLIMIT

IF (p > plimit) THEN
  pn1 = three * SQRT(p) * ((x / p) ** (one / three) + one /(nine * p) - one)
  fn_val = alnorm(pn1, .false.)
  RETURN
END IF

!      If X is extremely large compared to P then set fn_val = 1

IF (x > xbig) THEN
  fn_val = one
  RETURN
END IF

IF (x <= one .OR. x < p) THEN

!      Use Pearson's series expansion.
!      (Note that P is not large enough to force overflow in lngamma).
!      No need to test IFAULT on exit since P > 0.

  arg = p * LOG(x) - x - lngamma(p + one)
  c = one
  fn_val = one
  a = p
  40   a = a + one
  c = c * x / a
  fn_val = fn_val + c
  IF (c > tol) GO TO 40
  arg = arg + LOG(fn_val)
  fn_val = zero
  IF (arg >= elimit) fn_val = EXP(arg)

ELSE

!      Use a continued fraction expansion

  arg = p * LOG(x) - x - lngamma(p)
  a = one - p
  b = a + x + one
  c = zero
  pn1 = one
  pn2 = x
  pn3 = x + one
  pn4 = x * b
  fn_val = pn3 / pn4
  60   a = a + one
  b = b + two
  c = c + one
  an = a * c
  pn5 = b * pn3 - an * pn1
  pn6 = b * pn4 - an * pn2
  IF (ABS(pn6) > zero) THEN
    rn = pn5 / pn6
    IF (ABS(fn_val - rn) <= MIN(tol, tol * rn)) GO TO 80
    fn_val = rn
  END IF

  pn1 = pn3
  pn2 = pn4
  pn3 = pn5
  pn4 = pn6
  IF (ABS(pn5) >= oflo) THEN

!      Re-scale terms in continued fraction if terms are large

    pn1 = pn1 / oflo
    pn2 = pn2 / oflo
    pn3 = pn3 / oflo
    pn4 = pn4 / oflo
  END IF
  GO TO 60
  80   arg = arg + LOG(fn_val)
  fn_val = one
  IF (arg >= elimit) fn_val = one - EXP(arg)
END IF

RETURN
END FUNCTION gammad



FUNCTION alnorm(x, upper) RESULT(fn_val)

!  Algorithm AS66 Applied Statistics (1973) vol.22, no.3

!  Evaluates the tail area of the standardised normal curve
!  from x to infinity if upper is .true. or
!  from minus infinity to x if upper is .false.

! ELF90-compatible version by Alan Miller
! Latest revision - 29 November 1997

REAL (dp), INTENT(IN) :: x
LOGICAL, INTENT(IN)   :: upper
REAL (dp)             :: fn_val

! Local variables
REAL (dp), PARAMETER :: zero = 0.0_dp, one = 1.0_dp, half = 0.5_dp, &
                        con = 1.28_dp
REAL (dp) :: z, y
LOGICAL   :: up

!*** machine dependent constants
REAL (dp), PARAMETER :: ltone = 7.0_dp, utzero = 18.66_dp

REAL (dp), PARAMETER :: p = 0.398942280444_dp, q = 0.39990348504_dp,   &
                        r = 0.398942280385_dp, a1 = 5.75885480458_dp,  &
                        a2 = 2.62433121679_dp, a3 = 5.92885724438_dp,  &
                        b1 = -29.8213557807_dp, b2 = 48.6959930692_dp, &
                        c1 = -3.8052E-8_dp, c2 = 3.98064794E-4_dp,     &
                        c3 = -0.151679116635_dp, c4 = 4.8385912808_dp, &
                        c5 = 0.742380924027_dp, c6 = 3.99019417011_dp, &
                        d1 = 1.00000615302_dp, d2 = 1.98615381364_dp,  &
                        d3 = 5.29330324926_dp, d4 = -15.1508972451_dp, &
                        d5 = 30.789933034_dp

up = upper
z = x
IF(z >=  zero) GO TO 10
up = .NOT. up
z = -z
10 IF(z <= ltone .OR. up .AND. z <= utzero) GO TO 20
fn_val = zero
GO TO 40
20 y = half*z*z
IF(z > con) GO TO 30

fn_val = half - z*(p-q*y/(y+a1+b1/(y+a2+b2/(y+a3))))
GO TO 40
30 fn_val = r*EXP(-y)/(z+c1+d1/(z+c2+d2/(z+c3+d3/(z+c4+d4/(z+c5+d5/(z+c6))))))
40 IF(.NOT. up) fn_val = one - fn_val

RETURN
END FUNCTION alnorm



FUNCTION lngamma(z) RESULT(lanczos)

!  Uses Lanczos-type approximation to ln(gamma) for z > 0.
!  Reference:
!       Lanczos, C. 'A precision approximation of the gamma
!               function', J. SIAM Numer. Anal., B, 1, 86-96, 1964.
!  Accuracy: About 14 significant digits except for small regions
!            in the vicinity of 1 and 2.

!  Programmer: Alan Miller
!              1 Creswick Street, Brighton, Vic. 3187, Australia
!  Latest revision - 14 October 1996

REAL (dp), INTENT(IN) :: z
REAL (dp)             :: lanczos

! Local variables

REAL (dp), PARAMETER  :: a(9) =  &
                         (/ 0.9999999999995183_dp, 676.5203681218835_dp, &
                        -1259.139216722289_dp, 771.3234287757674_dp, &
                         -176.6150291498386_dp, 12.50734324009056_dp, &
                           -0.1385710331296526_dp, 0.9934937113930748D-05, &
                            0.1659470187408462D-06 /),  &
                         zero = 0._dp, one = 1._dp, half = 0.5_dp,   &
                         sixpt5 = 6.5_dp, seven = 7._dp,  &
                         lnsqrt2pi = 0.9189385332046727_dp

REAL (dp) :: tmp
INTEGER   :: j

IF (z <= zero) THEN
  WRITE(*, *) 'Error: zero or -ve argument for lngamma'
  RETURN
END IF

lanczos = zero
tmp = z + seven
DO j = 9, 2, -1
  lanczos = lanczos + a(j)/tmp
  tmp = tmp - one
END DO
lanczos = lanczos + a(1)
lanczos = LOG(lanczos) + lnsqrt2pi - (z + sixpt5) + (z - half)*LOG(z + sixpt5)
RETURN

END FUNCTION lngamma


END MODULE Fisher_Exact
