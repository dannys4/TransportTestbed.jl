function horner(x::Real, coeff::AbstractVector{<:Real})
    ret = coeff[1]
    for c in @view(coeff[2:end])
        ret = ret * x + c
    end
    ret
end

function HermiteCoeffs(N::Int)
    (N == 0) && return [1.0]
    (N == 1) && return [1.0, 0.0]
    N += 1
    coeffs_old = zeros(N + 2)
    coeffs_old[2] = 1.0
    coeffs_new = zeros(N + 2)
    for n in 2:N
        coeffs_new[1] = -1 * coeffs_old[2]
        for k in 1:n
            coeffs_new[k + 1] = coeffs_old[k] - (k + 1) * coeffs_old[k + 2]
        end
        tmp = coeffs_new
        coeffs_new = coeffs_old
        coeffs_old = tmp
    end
    return coeffs_new[(end - 2):-1:1]
end

function TestQuadrature(rng::AbstractRNG)
    # Make sure Horner works
    n_test_points = 1000
    x_test = randn(rng, n_test_points)
    for n_coeff_test in 1:5
        test_poly_coeff = randn(rng, n_coeff_test)
        test_poly = x -> sum(test_poly_coeff[end - j] * x^j for j in 0:(n_coeff_test - 1))
        test_horner = x -> horner(x, test_poly_coeff)
        test_poly_eval = test_poly.(x_test)
        test_horner_eval = test_horner.(x_test)
        @test norm(test_poly_eval - test_horner_eval) < 1e-8
    end

    quad_schemes = [(hermite_gk22, [1, 3, 9, 19, 41], [1, 5, 15, 29, 63], gaussprobhermite)]
    for (quad_scheme, point_counts, orders, ref_scheme) in quad_schemes
        for (n_pts, order) in zip(point_counts, orders)
            # Used mixed precision for checking; needed for high-degree polynomials
            test_coeffs = big.(HermiteCoeffs(order ÷ 2))
            test_horner_sq = x -> horner(x, test_coeffs)^2 / (factorial(big(order ÷ 2)))
            pts, wts = quad_scheme(n_pts)
            ref_pts, ref_wts = ref_scheme(order)
            test_approx = (test_horner_sq.(pts))'wts
            ref_approx = (test_horner_sq.(ref_pts))'ref_wts
            tol = 1e-8
            err = abs(test_approx - ref_approx)
            (abs(ref_approx) > 10) && (err /= abs(ref_approx))
            @test err<tol skip=(order > 50)
        end
    end
end
