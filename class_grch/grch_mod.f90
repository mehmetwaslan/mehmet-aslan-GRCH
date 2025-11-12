! grch_mod.f90
! Custom CLASS module for GRCH
! Compile with: make class

module grch
  use precision
  implicit none

  real(dl) :: grch_at, grch_Delta, grch_tau, grch_S0, grch_rho_mem_0

contains

  subroutine read_grch_params(pba)
    use io
    type(transfer) :: pba
    call io_read_real(pba%input,'grch_at',grch_at,0.65_dl)
    call io_read_real(pba%input,'grch_Delta',grch_Delta,0.30_dl)
    call io_read_real(pba%input,'grch_tau',grch_tau,0.0371_dl)
    call io_read_real(pba%input,'grch_S0',grch_S0,0.95_dl)
    call io_read_real(pba%input,'grch_rho_mem_0',grch_rho_mem_0,0.26_dl)
  end subroutine

  function w_grch(a)
    real(dl) :: w_grch, a
    w_grch = -0.5_dl * (1.0_dl + tanh((log(a) - log(grch_at)) / grch_Delta))
  end function

  function S_grch(a)
    real(dl) :: S_grch, a
    S_grch = grch_S0 + (1.0_dl - grch_S0) * (1.0_dl - tanh((log(a) - log(grch_at)) / 0.5_dl)) / 2.0_dl
  end function

end module grch
