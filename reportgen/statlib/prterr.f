      !-----------------------------------------------------------------------
      !  Name:       ERPRT

      !  Purpose:    Print an error message and stop.

      !  Usage:      CALL ERPRT (ICODE, MES)

      !  Arguments:
      !     ICODE  - Integer code for the error message.  (Input)
      !     MES    - Character string containing the error message.  (Input)
      !-----------------------------------------------------------------------
      subroutine prterr(icode, mes)
      !                                  SPECIFICATIONS FOR ARGUMENTS
      INTEGER, INTENT(IN)           :: icode
      CHARACTER (LEN=*), INTENT(IN) :: mes
cf2py intent(callback, hide) f2pystop
      external f2pystop

      CALL f2pystop(icode, mes)
      !WRITE (*, *) 'FEXACT ERROR: ', icode, ' ', mes
      !STOP

      END subroutine prterr