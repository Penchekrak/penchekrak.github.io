﻿(function () {
    try {
        (function () {
            function df(a, c, b, d) {
                var e = this;
                return D(window, "c.i", function () {
                    function f(w) {
                        (w = ef(k, l, "", w)(k, l)) && (T(w.then) ? w.then(g) : g(w));
                        return w
                    }

                    function g(w) {
                        w && (T(w) ? m.push(w) : La(w) && z(function (G) {
                            var Y = G[0];
                            G = G[1];
                            T(G) && ("u" === Y ? m.push(G) : h(G, Y))
                        }, pa(w)))
                    }

                    function h(w, G, Y) {
                        e[G] = El(k, l, Y || p, G, w)
                    }

                    var k = window;
                    (!k || isNaN(a) && !a) && ff();
                    var l = Fl(a, gf, c, b, d), m = [], p = [ph, ef, qh];
                    p.unshift(Gl);
                    var q = A(O, Cb), r = N(l);
                    l.id || Xa(ic("Invalid Metrika id: " + l.id, !0));
                    var v = $c.o("counters", {});
                    if (v[r]) return Db(k,
                        r, "Duplicate counter " + r + " initialization"), v[r];
                    v[r] = e;
                    $c.C("counters", v);
                    $c.Va("counter", e);
                    z(function (w) {
                        w(k, l)
                    }, Xd);
                    z(f, Cc);
                    f(Hl);
                    h(Il(k, l, m), "destruct", [ph, qh]);
                    Kb(k, E([k, q, f, 1, "a.i"], rh));
                    z(f, U)
                })()
            }

            function Jl(a, c) {
                var b;
                try {
                    var d = c.origin
                } catch (f) {
                }
                if ("https://oauth.yandex.ru" === d && n(c, "source.window") && "_ym_uid_request" === n(c.data, "_ym")) {
                    d = c.source;
                    var e = (b = {}, b._ym_uid = a(), b);
                    d.postMessage(e, "https://oauth.yandex.ru")
                }
            }

            function Kl(a) {
                var c = Z(Boolean, A(function (b) {
                    var d = b[1];
                    return (b = Ll(a[b[0]])) ?
                        "" + d + "\n" + b : null
                }, pa(Ml)));
                return J("\n", c)
            }

            function Nl(a) {
                return "che\n" + a
            }

            function Ll(a) {
                return wa(a) ? a : ca(a) ? J(",", A(function (c) {
                    return '"' + c.brand + '";v="' + c.version + '"'
                }, a)) : ma(a) ? "" : a ? "?1" : "?0"
            }

            function Ol(a, c) {
                var b = Pl(a), d = [Ql(a) || Rl(a)];
                sh(a) && d.push(b);
                var e = fa(a);
                b = Ra(a);
                var f = b.o("synced", {});
                d = Z(function (g) {
                    if (c[g]) {
                        var h = (f[g] || 1) + 1440 < e(lb);
                        h && delete f[g];
                        return h
                    }
                }, d);
                b.C("synced", f);
                return A(function (g) {
                    return {Fj: c[g], Ri: g}
                }, d)
            }

            function Rl(a) {
                a = Sl(a);
                return Tl[a] || a
            }

            function Pl(a) {
                a =
                    th(a);
                return Ul[a] || "ru"
            }

            function Vl(a, c) {
                var b = "" + c, d = {id: 1, ba: "0"}, e = Wl(b);
                e ? d.id = e : -1 === b.indexOf(":") ? (b = Ga(b), d.id = b) : (b = b.split(":"), e = b[1], d.id = Ga(b[0]), d.ba = Yd(e) ? "1" : "0");
                return [Ha(a, d), d]
            }

            function Xl(a, c) {
                ad[a] || (ad[a] = []);
                ad[a].push(c)
            }

            function Yl(a, c, b, d) {
                var e = b.H;
                if (c.ak || Yd(c.ba) || !e) d(); else {
                    var f = Zd(a), g = Dc(a, ""), h = function () {
                        var v = uh(f);
                        v = "" + v + Zl(v, g);
                        $d(b, "gdpr", v);
                        d()
                    };
                    if (3 === c.id) h(); else {
                        var k = H(a);
                        if (e = k.o("f1")) e(h); else {
                            var l = (e = uh(f)) ? A(u(ae, n), e.split(",")) : [];
                            if (vh(l)) h();
                            else {
                                e = be(a);
                                var m = S(a);
                                var p = /(^|\w+\.)yango(\.yandex)?\.com$/.test(m.hostname) ? {
                                    url: "https://yastatic.net/s3/taxi-front/yango-gdpr-popup/",
                                    Sf: "ar az be en es et fi fr he hy ka kk ky lt lv no pt ro ru sl sr tg tr uk uz zh".split(" "),
                                    ag: "_inversed_buttons"
                                } : void 0;
                                var q = (e = e || !!p) && (-1 !== m.href.indexOf("yagdprcheck=1") || g.o("yaGdprCheck"));
                                m = g.o("gdpr");
                                var r = I.resolve();
                                g.o("yandex_login") ? (l.push("13"), g.C("gdpr", Ec, 525600)) : e ? K(m, Xb) ? m === hf ? l.push("12") : l.push("3") : jf(a) || $l(a) ? l.push("17") : r =
                                    am(a).then(function (v) {
                                        v && l.push("28")
                                    }, B) : l.push("14");
                                r.then(function () {
                                    var v = u(f, bm);
                                    vh(l) ? (z(v, l), h()) : (ce.push(h), k.C("f1", function (w, G) {
                                        var Y = 0;
                                        if (G) {
                                            var Q = mb(a, G) || "";
                                            Y += Q.length
                                        }
                                        ce.push(w);
                                        1E6 >= Y && ce.push(w)
                                    }), (0, kf[0])(a).then(X("params.eu")).then(function (w) {
                                        if (w || q) {
                                            g.C("gdpr_popup", hf);
                                            cm(a, c);
                                            if (eb(a)) return dm(a, v, c);
                                            var G = wh(a, f);
                                            if (G) return w = em(a, v, G, c, p), w.then(E([a, c], fm)), w
                                        }
                                        w || v("8");
                                        return I.resolve({value: Ec, ce: !0})
                                    }).then(function (w) {
                                        g.Rb("gdpr_popup");
                                        if (w) {
                                            var G = w.value;
                                            w =
                                                w.ce;
                                            K(G, Xb) && g.C("gdpr", G, w ? void 0 : 525600)
                                        }
                                        G = jc(ce, ha);
                                        kc(a, G, 20)(Sa(D(a, "gdr"), B));
                                        k.C("f1", ha)
                                    })["catch"](D(a, "gdp.a")))
                                })
                            }
                        }
                    }
                }
            }

            function fm(a, c) {
                if (be(a)) {
                    var b = Zd(a), d = Ha(a, c);
                    d = d && d.params;
                    b = A(u(gm, n), lf(b));
                    d && b.length && d("gdpr", Na(b))
                }
            }

            function dm(a, c, b) {
                var d = de(a, b);
                return new I(function (e) {
                    var f;
                    if (d) {
                        var g = d.ca, h = t(u("4", c), u(null, e)), k = V(a, h, 2E3, "gdp.f.t");
                        d.sg((f = {}, f.type = "isYandex", f)).then(function (l) {
                            l.isYandex ? (c("5"), g.D(xa(["GDPR-ok-view-default", "GDPR-ok-view-detailed"], mf), function (m) {
                                    e({value: xh(m[1].type)})
                                })) :
                                (c("6"), e(null))
                        })["catch"](h).then(E([a, k], na))
                    } else e({value: hf, ce: !0})
                })
            }

            function cm(a, c) {
                var b = de(a, c);
                b && b.ca.D(["isYandex"], function () {
                    var d;
                    return d = {type: "isYandex"}, d.isYandex = be(a), d
                });
                return b
            }

            function em(a, c, b, d, e) {
                var f = void 0 === e ? hm : e;
                e = f.url;
                var g = f.ag;
                f = im(a, f.Sf, d.bk);
                var h = de(a, d);
                if (!h) return I.resolve({value: Ec, ce: !0});
                var k = lc(a, {src: "" + e + f + g + ".js"});
                return new I(function (l, m) {
                    k ? (c("7"), k.onerror = function () {
                        var p;
                        c("9");
                        h.rg((p = {}, p.type = "GDPR-ok-view-default", p));
                        l(null)
                    }, k.onload =
                        function () {
                            c("10");
                            b.D(xa(["GDPR-ok-view-default", "GDPR-ok-view-detailed"], mf), function (p) {
                                var q;
                                p = p.type;
                                h.rg((q = {}, q.type = p, q));
                                l({value: xh(p)})
                            })
                        }) : (c("9"), m(Ta("gdp.e")))
                })
            }

            function im(a, c, b) {
                a = b || th(a);
                return K(a, c) ? a : "en"
            }

            function xh(a) {
                if (K(a, ["GDPR-ok-view-default", "GDPR-ok-view-detailed"])) return Ec;
                a = a.replace("GDPR-ok-view-detailed-", "");
                return K(a, Xb) ? a : Ec
            }

            function yh(a, c, b) {
                var d = n(a, "AppMetricaInitializer"), e = n(d, "init");
                if (e) try {
                    F(e, d)(mb(a, c))
                } catch (f) {
                } else zh = V(a, E([a, c, 2 * b], yh),
                    b, "ai.d");
                return function () {
                    return na(a, zh)
                }
            }

            function jm(a) {
                var c = n(a, "speechSynthesis.getVoices");
                if (!c) return "";
                a = F(c, a.speechSynthesis);
                return mc(function (b) {
                    return A(u(b, n), km)
                }, a())
            }

            function lm(a, c, b) {
                return J("x", A(t(O, qa("concat", "" + a), u(b, n)), c))
            }

            function mm(a, c) {
                var b = c.oh;
                if (!nm(a, b)) return "";
                var d = [];
                a:{
                    var e = om(a, b);
                    try {
                        var f = E(e, t)()();
                        break a
                    } catch (G) {
                        if ("ccf" === G.message) {
                            f = null;
                            break a
                        }
                        Xa(G)
                    }
                    f = void 0
                }
                if (Ua(f)) var g = ""; else try {
                    g = f.toDataURL()
                } catch (G) {
                    g = ""
                }
                (f = g) && d.push(f);
                var h = b.getContextAttributes();
                try {
                    var k = Ma(b.getSupportedExtensions, "getSupportedExtensions") ? b.getSupportedExtensions() || [] : []
                } catch (G) {
                    k = []
                }
                k = J(";", k);
                f = nf(b.getParameter(b.ALIASED_LINE_WIDTH_RANGE), b);
                e = nf(b.getParameter(b.ALIASED_POINT_SIZE_RANGE), b);
                g = b.getParameter(b.ALPHA_BITS);
                h = h && h.antialias ? "yes" : "no";
                var l = b.getParameter(b.BLUE_BITS), m = b.getParameter(b.DEPTH_BITS), p = b.getParameter(b.GREEN_BITS),
                    q = b.getExtension("EXT_texture_filter_anisotropic") || b.getExtension("WEBKIT_EXT_texture_filter_anisotropic") || b.getExtension("MOZ_EXT_texture_filter_anisotropic");
                if (q) {
                    var r = b.getParameter(q.MAX_TEXTURE_MAX_ANISOTROPY_EXT);
                    0 === r && (r = 2)
                }
                r = {
                    rk: k,
                    "webgl aliased line width range": f,
                    "webgl aliased point size range": e,
                    "webgl alpha bits": g,
                    "webgl antialiasing": h,
                    "webgl blue bits": l,
                    "webgl depth bits": m,
                    "webgl green bits": p,
                    "webgl max anisotropy": q ? r : null,
                    "webgl max combined texture image units": b.getParameter(b.MAX_COMBINED_TEXTURE_IMAGE_UNITS),
                    "webgl max cube map texture size": b.getParameter(b.MAX_CUBE_MAP_TEXTURE_SIZE),
                    "webgl max fragment uniform vectors": b.getParameter(b.MAX_FRAGMENT_UNIFORM_VECTORS),
                    "webgl max render buffer size": b.getParameter(b.MAX_RENDERBUFFER_SIZE),
                    "webgl max texture image units": b.getParameter(b.MAX_TEXTURE_IMAGE_UNITS),
                    "webgl max texture size": b.getParameter(b.MAX_TEXTURE_SIZE),
                    "webgl max varying vectors": b.getParameter(b.MAX_VARYING_VECTORS),
                    "webgl max vertex attribs": b.getParameter(b.MAX_VERTEX_ATTRIBS),
                    "webgl max vertex texture image units": b.getParameter(b.MAX_VERTEX_TEXTURE_IMAGE_UNITS),
                    "webgl max vertex uniform vectors": b.getParameter(b.MAX_VERTEX_UNIFORM_VECTORS),
                    "webgl max viewport dims": nf(b.getParameter(b.MAX_VIEWPORT_DIMS), b),
                    "webgl red bits": b.getParameter(b.RED_BITS),
                    "webgl renderer": b.getParameter(b.RENDERER),
                    "webgl shading language version": b.getParameter(b.SHADING_LANGUAGE_VERSION),
                    "webgl stencil bits": b.getParameter(b.STENCIL_BITS),
                    "webgl vendor": b.getParameter(b.VENDOR),
                    "webgl version": b.getParameter(b.VERSION)
                };
                of(d, r, ": ");
                a:{
                    try {
                        var v = b.getExtension("WEBGL_debug_renderer_info");
                        if (v) {
                            var w = {
                                "webgl unmasked vendor": b.getParameter(v.UNMASKED_VENDOR_WEBGL),
                                "webgl unmasked renderer": b.getParameter(v.UNMASKED_RENDERER_WEBGL)
                            };
                            break a
                        }
                    } catch (G) {
                    }
                    w = {}
                }
                of(d, w);
                if (!b.getShaderPrecisionFormat) return J("~", d);
                of(d, pm(b));
                return J("~", d)
            }

            function of(a, c, b) {
                void 0 === b && (b = ":");
                z(function (d) {
                    return a.push("" + d[0] + b + d[1])
                }, pa(c))
            }

            function qm(a, c, b, d) {
                c = d.o("cc");
                d = E(["cc", ""], d.C);
                if (c) {
                    var e = c.split("&");
                    c = e[0];
                    if ((e = (e = e[1]) && Ga(e)) && 1440 < fa(a)(lb) - e) return d();
                    b.C("cc", c)
                } else ka(0)(c) || d()
            }

            function rm(a, c, b, d) {
                return ra(c, function (e) {
                    if ("0" === n(e, "settings.pcs") &&
                        !bd(a)) if (e = d.o("zzlc"), W(e) || Ua(e) || "na" === e) {
                        e = "ru";
                        var f = pf(a, 68), g = qf(a, 79);
                        if (f || g) e = "md";
                        if (f = ab(a)) {
                            var h = f("iframe");
                            y(h.style, {display: "none", width: "1px", height: "1px", visibility: "hidden"});
                            h.src = "https://mc.yandex." + e + Ah("L21ldHJpa2EvenpsYy5odG1s");
                            if (e = Yb(a)) {
                                e.appendChild(h);
                                var k = 0, l = ia(a).D(a, ["message"], D(a, "zz.m", function (m) {
                                    (m = n(m, "data")) && m.substr && "__ym__zz" === m.substr(0, 8) && (nc(h), m = m.substr(8), d.C("zzlc", m), b.C("zzlc", m), l(), na(a, k))
                                }));
                                k = V(a, t(l, u(h, nc)), 3E3)
                            }
                        }
                    } else b.C("zzlc",
                        e)
                })
            }

            function sm(a, c, b) {
                var d, e;
                c = bb(u(a, n), tm);
                c = W(c) ? null : n(a, c);
                if (n(a, "navigator.onLine") && c && c && n(c, "prototype.constructor.name")) {
                    var f = new c((d = {}, d.iceServers = [], d));
                    a = n(f, "createDataChannel");
                    T(a) && (F(a, f, "y.metrika")(), a = n(f, "createOffer"), T(a) && !a.length && (a = F(a, f)(), d = n(a, "then"), T(d) && F(d, a, function (g) {
                        var h = n(f, "setLocalDescription");
                        T(h) && F(h, f, g, B, B)()
                    })(), y(f, (e = {}, e.onicecandidate = function () {
                        var g, h = n(f, "close");
                        if (T(h)) {
                            h = F(h, f);
                            try {
                                var k = (g = n(f, "localDescription.sdp")) && g.match(/c=IN\s[\w\d]+\s([\w\d:.]+)/)
                            } catch (l) {
                                f.onicecandidate =
                                    B;
                                "closed" !== f.iceConnectionState && h();
                                return
                            }
                            k && 0 < k.length && (g = oc(k[1]), b.C("pp", g));
                            f.onicecandidate = B;
                            h()
                        }
                    }, e))))
                }
            }

            function um(a, c, b) {
                var d, e = cd(a, c);
                if (e) {
                    e.ca.D(["gpu-get"], function () {
                        var h;
                        return h = {}, h.type = "gpu-get", h.pu = b.o("pu"), h
                    });
                    var f = n(a, "opener");
                    if (f) {
                        var g = V(a, E([a, c, b], Bh), 200, "pu.m");
                        e.Je(f, (d = {}, d.type = "gpu-get", d), function (h, k) {
                            var l = n(k, "pu");
                            l && (na(a, g), b.C("pu", l))
                        })
                    } else Bh(a, c, b)
                }
            }

            function Bh(a, c, b) {
                var d = n(a, "location.host");
                a = dd(a, c);
                b.C("pu", "" + oc(d) + a)
            }

            function Ch(a,
                        c, b) {
                c = Dc(a, void 0, c);
                c = Dh(a, c.o("phc_settings") || "");
                var d = n(c, "clientId"), e = n(c, "orderId"), f = n(c, "service_id"), g = n(c, "phones") || [];
                return d && e && g ? vm(a, b.xc, {Gg: wm}).fg(g).then(function (h) {
                    return xm(b, {Pb: d, $b: e, wg: f}, h.qa, g, h.Ba)
                })["catch"](function () {
                }) : I.resolve()
            }

            function wm(a, c, b) {
                a = zm(b.dd);
                if ("href" === b.Ge) {
                    var d = b.Db;
                    c = d.href;
                    b = c.replace(a, b.ib);
                    if (c !== b) return d.href = b, !0
                } else if ((a = null === (d = b.Db.textContent) || void 0 === d ? void 0 : d.replace(a, b.ib)) && a !== b.Db.textContent) return b.Db.textContent =
                    a, !0;
                return !1
            }

            function xm(a, c, b, d, e) {
                var f;
                c.Pb && c.$b && (c.Pb === a.Pb && c.$b === a.$b || y(a, c, {
                    qa: {},
                    ob: !0
                }), 0 < e && Oa(a.Ba, [e]), z(function (g) {
                    var h, k, l = g[0];
                    g = g[1];
                    var m = +(a.qa[l] && a.qa[l][g] ? a.qa[l][g] : 0);
                    y(a.qa, (h = {}, h[l] = (k = {}, k[g] = m, k), h))
                }, d), z(function (g) {
                    var h, k, l = g[0];
                    g = g[1];
                    var m = 1 + (a.qa[l] ? a.qa[l][g] : 0);
                    y(a.qa, (h = {}, h[l] = (k = {}, k[g] = m, k), h))
                }, b), a.Pf && (a.ob || b.length) && ((c = Ha(a.l, a.xc)) && c.params("__ym", "phc", (f = {}, f.clientId = a.Pb, f.orderId = a.$b, f.service_id = a.wg, f.phones = a.qa, f.performance = a.Ba,
                    f)), a.ob = !1))
            }

            function Am(a) {
                a = ab(a);
                if (!a) return "";
                a = a("video");
                try {
                    var c = qa("canPlayType", a), b = mc(function (d) {
                        return A(t(O, qa("concat", d + "; codecs=")), Bm)
                    }, Eh);
                    return A(c, [].concat(Eh, b))
                } catch (d) {
                    return "canPlayType"
                }
            }

            function Cm(a) {
                var c = n(a, "matchMedia");
                if (c && Aa("matchMedia", c)) {
                    var b = qa("matchMedia", a);
                    return M(function (d, e) {
                        d[e] = b("(" + e + ")");
                        return d
                    }, {}, Dm)
                }
            }

            function pm(a) {
                return M(function (c, b) {
                    var d = b[0], e = b[1];
                    c[d + " precision"] = n(e, "precision") || "n";
                    c[d + " precision rangeMin"] = n(e, "rangeMin") ||
                        "n";
                    c[d + " precision rangeMax"] = n(e, "rangeMax") || "n";
                    return c
                }, {}, [["webgl vertex shader high float", a.getShaderPrecisionFormat(a.VERTEX_SHADER, a.HIGH_FLOAT)], ["webgl vertex shader medium", a.getShaderPrecisionFormat(a.VERTEX_SHADER, a.MEDIUM_FLOAT)], ["webgl vertex shader low float", a.getShaderPrecisionFormat(a.VERTEX_SHADER, a.LOW_FLOAT)], ["webgl fragment shader high float", a.getShaderPrecisionFormat(a.FRAGMENT_SHADER, a.HIGH_FLOAT)], ["webgl fragment shader medium float", a.getShaderPrecisionFormat(a.FRAGMENT_SHADER,
                    a.MEDIUM_FLOAT)], ["webgl fragment shader low float", a.getShaderPrecisionFormat(a.FRAGMENT_SHADER, a.LOW_FLOAT)], ["webgl vertex shader high int", a.getShaderPrecisionFormat(a.VERTEX_SHADER, a.HIGH_INT)], ["webgl vertex shader medium int", a.getShaderPrecisionFormat(a.VERTEX_SHADER, a.MEDIUM_INT)], ["webgl vertex shader low int", a.getShaderPrecisionFormat(a.VERTEX_SHADER, a.LOW_INT)], ["webgl fragment shader high int", a.getShaderPrecisionFormat(a.FRAGMENT_SHADER, a.HIGH_INT)], ["webgl fragment shader medium int",
                    a.getShaderPrecisionFormat(a.FRAGMENT_SHADER, a.MEDIUM_INT)], ["webgl fragment shader low int precision", a.getShaderPrecisionFormat(a.FRAGMENT_SHADER, a.LOW_INT)]])
            }

            function om(a, c) {
                return [function () {
                    var b = c.createBuffer();
                    b && c.getParameter && Aa("getParameter", c.getParameter) || rf();
                    c.bindBuffer(c.ARRAY_BUFFER, b);
                    var d = new a.Float32Array(Em);
                    c.bufferData(c.ARRAY_BUFFER, d, c.STATIC_DRAW);
                    b.Ni = 3;
                    b.Zi = 3;
                    d = c.createProgram();
                    var e = c.createShader(c.VERTEX_SHADER);
                    d && e || rf();
                    return {Ee: d, Zj: e, Yj: b}
                }, function (b) {
                    var d =
                        b.Ee, e = b.Zj;
                    c.shaderSource(e, "attribute vec2 attrVertex;varying vec2 varyinTexCoordinate;uniform vec2 uniformOffset;void main(){varyinTexCoordinate=attrVertex+uniformOffset;gl_Position=vec4(attrVertex,0,1);}");
                    c.compileShader(e);
                    c.attachShader(d, e);
                    (d = c.createShader(c.FRAGMENT_SHADER)) || rf();
                    return y(b, {Wh: d})
                }, function (b) {
                    var d = b.Ee, e = b.Wh;
                    c.shaderSource(e, "precision mediump float;varying vec2 varyinTexCoordinate;void main() {gl_FragColor=vec4(varyinTexCoordinate,0,1);}");
                    c.compileShader(e);
                    c.attachShader(d,
                        e);
                    c.linkProgram(d);
                    c.useProgram(d);
                    return b
                }, function (b) {
                    var d = b.Ee;
                    b = b.Yj;
                    d.Xj = c.getAttribLocation(d, "attrVertex");
                    d.aj = c.getUniformLocation(d, "uniformOffset");
                    c.enableVertexAttribArray(d.Tk);
                    c.vertexAttribPointer(d.Xj, b.Ni, c.FLOAT, !1, 0, 0);
                    c.uniform2f(d.aj, 1, 1);
                    c.drawArrays(c.TRIANGLE_STRIP, 0, b.Zi);
                    return c.canvas
                }]
            }

            function nm(a, c) {
                if (!T(a.Float32Array)) return !1;
                var b = n(c, "canvas");
                if (!b || !Aa("toDataUrl", b.toDataURL)) return !1;
                try {
                    c.createBuffer()
                } catch (d) {
                    return !1
                }
                return !0
            }

            function nf(a, c) {
                c.clearColor(0,
                    0, 0, 1);
                c.enable(c.DEPTH_TEST);
                c.depthFunc(c.LEQUAL);
                c.clear(c.COLOR_BUFFER_BIT | c.DEPTH_BUFFER_BIT);
                return "[" + n(a, "0") + ", " + n(a, "1") + "]"
            }

            function Fm(a, c, b) {
                function d(q) {
                    return function () {
                        var r = b.o("scip", "") + q;
                        b.C("scip", r)
                    }
                }

                var e, f = ed(a, "ci");
                f = Eb(a, f);
                var g = sf(a), h = fa(a)(lb), k = ["sync.cook.int"], l = Fh(g.o("sci"));
                if (!l || 1440 < h - l) {
                    b.C("scip", "0");
                    var m = d("a"), p = Fc(a, c);
                    return f({
                        Y: {Fa: k, Ib: 1500, qd: !0},
                        G: (e = {}, e.duid = p, e.hid = "" + Lb(a), e)
                    }, ["https://an.yandex.ru/sync_cookie"]).then(function (q) {
                        q = n(q.fd,
                            "CookieMatchUrls");
                        ca(q) || (q = []);
                        Pa(q) ? d("1")() : m();
                        var r = ed(a, "c"), v = Eb(a, r);
                        q = A(function (w, G) {
                            var Y = "" + w + (pc(w, "?") ? "&" : "?") + "duid=" + p;
                            return v({Y: {Fa: k, Ib: 1500}}, ["https://" + Y]).then(B, t(d("b"), d("" + G)))
                        }, Z(wa, q));
                        return I.all(q)
                    }, m).then(function () {
                        var q = b.o("scip");
                        !q || pc(q, "a") || pc(q, "b") || (g.C("sci", h), d("2")())
                    }, B)
                }
                return I.resolve()
            }

            function Gm() {
                return M(function (a, c) {
                    var b = oc(c + "/tag.js");
                    Gh[b] || (a[b] = 1);
                    return a
                }, {}, ["mc.yandex.ru", "mc.yandex.com", "cdn.jsdelivr.net/npm/yandex-metrica-watch"])
            }

            function Hh(a) {
                return {
                    Z: function (c, b) {
                        if (!c.H) return b();
                        var d = H(a).o("fid");
                        !Ih && d && ($d(c, "fid", d), Ih = !0);
                        return b()
                    }
                }
            }

            function Hm(a, c) {
                var b = a.document;
                if (K(b.readyState, ["interactive", "complete"])) Kb(a, c); else {
                    var d = ia(a), e = d.D, f = d.kc, g = function () {
                        f(b, ["DOMContentLoaded"], g);
                        f(a, ["load"], g);
                        c()
                    };
                    e(b, ["DOMContentLoaded"], g);
                    e(a, ["load"], g)
                }
            }

            function tf(a) {
                return {
                    Z: function (c, b) {
                        var d = c.H;
                        if (d) {
                            var e = H(a).o("adBlockEnabled");
                            e && d.C("adb", e)
                        }
                        b()
                    }
                }
            }

            function Im(a) {
                var c = D(a, "i.clch", Jm);
                ia(a).D(a.document,
                    ["click"], F(c, null, a), {passive: !1});
                return function (b) {
                    var d = sa.Za, e = a.Ya[sa.uc], f = !!e._informer;
                    e._informer = y({domain: "informer.yandex.ru"}, b);
                    f || lc(a, {src: d + "//informer.yandex.ru/metrika/informer.js"})
                }
            }

            function Km(a, c) {
                var b = Ra(a);
                if ("" === b.o("cc")) {
                    var d = u("cc", b.C);
                    d(0);
                    var e = fa(a), f = H(a);
                    f = t(X(fd({fd: 1}) + ".c"), gd(function (g) {
                        d(g + "&" + e(lb))
                    }), u("cc", f.C));
                    Ba(a, "6", c)({
                        Y: {
                            qd: !0,
                            Pg: !1
                        }
                    }, ["https://mc.yandex.md/cc"]).then(f)["catch"](t(gd(function () {
                        var g = e(lb);
                        b.C("cc", "&" + g)
                    }), D(a, "cc")))
                }
            }

            function ee(a,
                        c) {
                if (!c) return !1;
                var b = S(a);
                return (new RegExp(c)).test("" + b.pathname + b.hash + b.search)
            }

            function Lm(a, c) {
                return ra(c, function (b) {
                    var d = n(b, "settings.dr");
                    return {Eh: Mm(a, d), isEnabled: n(b, "settings.auto_goals")}
                })
            }

            function Nm(a, c, b, d, e) {
                b = uf(a.document.body, b);
                d = uf(a.document.body, d);
                K(e.target, [b, d]) && vf(a, c)
            }

            function Jh(a, c, b, d) {
                (b = Om(a, d, b)) && vf(a, c, b)
            }

            function Kh(a, c) {
                var b = Lh(a, c);
                return Pm(a, b)
            }

            function Lh(a, c) {
                var b = uf(a.document.body, c);
                return b ? Qm(a, b) : ""
            }

            function vf(a, c, b) {
                if (c = Ha(a, c)) a = Gc(["dr",
                    b || "" + Va(a, 10, 99)]), c.params(Gc(["__ym", a]))
            }

            function uf(a, c) {
                var b = null;
                try {
                    b = c ? qc(c, a) : b
                } catch (d) {
                }
                return b
            }

            function Mh(a) {
                a = ya(Ah(a));
                return A(function (c) {
                    c = c.charCodeAt(0).toString(2);
                    return Nh("0", 8, c)
                }, a)
            }

            function Qm(a, c) {
                if (!c) return "";
                var b = [], d = n(a, "document");
                wf(a, c, function (e) {
                    if (e.nodeType === d.TEXT_NODE) var f = e.textContent; else e instanceof a.HTMLImageElement ? f = e.alt : e instanceof a.HTMLInputElement && (f = e.value);
                    (f = f && f.trim()) && b.push(f)
                });
                return 0 === b.length ? "" : b.join(" ")
            }

            function Rm(a,
                        c, b) {
                a = Ca(b);
                b = a[1];
                "track" === a[0] && c({version: "0", Dc: b})
            }

            function Sm(a, c, b) {
                if (b) {
                    var d = b.version;
                    (b = n(Tm, d + "." + b.Dc)) && (c && K(b, Um) || a("ym-" + b + "-" + d))
                }
            }

            function Vm(a, c, b) {
                var d, e = Oh(a, c), f = S(a);
                f = fe(f.protocol + "//" + f.hostname + f.pathname);
                c = dd(a, c);
                var g = "";
                do g += Va(a); while (g.length < c.length);
                g = g.slice(0, c.length);
                a = "";
                for (var h = 0; h < c.length; h += 1) a += (c.charCodeAt(h) + g.charCodeAt(h) - 96) % 10;
                c = [g, a];
                a = c[0];
                c = c[1];
                return (d = {}, d.mf = "https://adstat.yandex.ru/track?service=metrika&id=" + c + "&mask=" + a + "&ref=" +
                    f, d.rt = "https://" + e + ".mc.yandex.ru/watch/3/1?browser-info=rt:1", d)[b]
            }

            function Wm(a, c, b) {
                var d = n(b, "data");
                if (wa(d)) {
                    var e = d.split("*");
                    d = e[0];
                    var f = e[1];
                    e = e[2];
                    "sc.frame" === d ? b.source.postMessage("sc.images*" + a, "*") : "sc.image" === d && f === a && c(e)
                }
            }

            function Xm(a, c) {
                var b = Ra(a), d = "wv2rf:" + N(c), e = c.sc, f = xf(a), g = b.o(d), h = c.Qj;
                return W(f) || Ua(g) ? za(function (k, l) {
                        ra(c, function (m) {
                            var p = n(m, "settings.webvisor.forms");
                            p = !n(m, "settings.x3") && p;
                            f = xf(a) || n(m, "settings.eu");
                            b.C(d, hd(p));
                            l({sc: e, ae: !!f, dg: p, Ig: h})
                        })
                    }) :
                    yf({sc: e, ae: f, dg: !!Ga(g), Ig: h})
            }

            function Ym() {
                var a = M(function (c, b) {
                    c[b[0]] = {xd: 0, mh: 1 / b[1]};
                    return c
                }, {}, [["blur", .0034], ["change", .0155], ["click", .01095], ["deviceRotation", 2E-4], ["focus", .0061], ["mousemove", .5132], ["scroll", .4795], ["selection", .0109], ["touchcancel", 2E-4], ["touchend", .0265], ["touchforcechange", .0233], ["touchmove", .1442], ["touchstart", .027], ["zoom", .0014]]);
                return {
                    fh: function (c) {
                        if (c.length) return {
                            type: "activity", data: M(function (b, d) {
                                var e = a[d];
                                return Math.round(b + e.xd * e.mh)
                            }, 0, da(a))
                        }
                    },
                    bj: function (c) {
                        c && (c = a[c.data.type]) && (c.xd += 1)
                    }
                }
            }

            function Zm(a) {
                if (a.type && a.event) switch (a.type) {
                    case "page":
                        var c = a.data, b = c.Ka, d = c.ld, e = c.content;
                        delete c.Ka;
                        delete c.ld;
                        delete c.content;
                        c = {type: "page", data: {M: a.M || 0, content: e, Ka: b, ld: d, aa: c}};
                        a.L && (c.L = a.L);
                        return c;
                    case "event":
                        a:{
                            c = {type: "event", L: a.L, data: {M: a.M, type: a.event, aa: {}}};
                            b = y({}, a.data);
                            switch (a.event) {
                                case "zoom":
                                    c.data.aa = {Qg: {x: 0, y: 0, level: 0}, Sg: b, duration: 1};
                                    break;
                                case "keystroke":
                                    c.data.aa = b.Sc;
                                    break;
                                case "deviceRotation":
                                case "resize":
                                    c.data.aa =
                                        b;
                                    break;
                                case "windowfocus":
                                case "windowblur":
                                case "focus":
                                    c.data.target = b.target;
                                    c.data.aa = null;
                                    break;
                                case "touchmove":
                                case "touchstart":
                                case "touchend":
                                case "touchcancel":
                                case "touchforcechange":
                                case "scroll":
                                case "change":
                                case "click":
                                case "mousemove":
                                case "mousedown":
                                case "mouseup":
                                case "selection":
                                case "stylechange":
                                    c.data.target = b.target;
                                    delete b.target;
                                    c.data.aa = b;
                                    break;
                                case "srcset":
                                    a = {
                                        type: "mutation",
                                        L: a.L,
                                        data: {
                                            M: a.M,
                                            aa: {
                                                Oa: [{
                                                    wd: [{
                                                        id: b.target,
                                                        Ob: {src: {Wc: "", n: b.value, r: !1}},
                                                        Ha: 0
                                                    }]
                                                }], index: 0
                                            }
                                        }
                                    };
                                    break a
                            }
                            a = c
                        }
                        break;
                    case "mutation":
                        return $m(a)
                }
                return a
            }

            function $m(a) {
                var c = y({}, a.data), b = [];
                switch (a.event) {
                    case "tc":
                        b = [{d: [{id: c.target, Ac: {Wc: "", n: c.value}, Ha: c.index}]}];
                        break;
                    case "ac":
                        b = [{
                            wd: [{
                                id: c.target, Ob: M(function (d, e) {
                                    var f = e[1];
                                    d[e[0]] = {Wc: "", n: f, r: ma(f)};
                                    return d
                                }, {}, pa(c.attributes)), Ha: c.index
                            }]
                        }];
                        break;
                    case "re":
                        b = [{
                            cf: A(function (d) {
                                return {id: d, Ha: c.index}
                            }, c.na)
                        }];
                        break;
                    case "ad":
                        b = [{
                            gf: A(function (d) {
                                return {
                                    id: d.id,
                                    Wf: d.name,
                                    Xf: d.Vf,
                                    ue: d.parent,
                                    xe: d.ze,
                                    ne: d.next,
                                    Ob: d.attributes,
                                    Ha: c.index,
                                    Ac: d.content,
                                    Qa: d.hidden
                                }
                            }, c.na)
                        }]
                }
                return {type: "mutation", L: a.L, data: {M: a.M, aa: {Oa: b, index: c.index}}}
            }

            function an(a) {
                return {
                    Yh: function () {
                        var c = a.document.querySelector("base[href]");
                        return c ? c.getAttribute("href") : null
                    }, $h: function () {
                        if (a.document.doctype) {
                            var c = y({name: "html", publicId: "", systemId: ""}, a.document.doctype), b = c.publicId,
                                d = c.systemId;
                            return "<!DOCTYPE " + J("", [c.name, b ? ' PUBLIC "' + b + '"' : "", !b && d ? " SYSTEM" : "", d ? ' "' + d + '"' : ""]) + ">"
                        }
                        return null
                    }, mi: function () {
                        try {
                            if (!a.sessionStorage) return null;
                            var c = a.sessionStorage.getItem("__vw_tab_guid"), b = !1;
                            try {
                                var d = a.opener ? a.opener.sessionStorage : null;
                                b = !!d && c === d.getItem("__vw_tab_guid")
                            } catch (e) {
                                b = !0
                            }
                            if (!c || b) c = Ph(), a.sessionStorage.setItem("__vw_tab_guid", c);
                            return c
                        } catch (e) {
                            return null
                        }
                    }
                }
            }

            function bn(a, c, b) {
                var d = id(a), e = ia(a), f = eb(a), g = c.Od(),
                    h = !n(a, "postMessage") || f && !n(a, "parent.postMessage"), k = u(d, O);
                if (h) {
                    if (!g) return V(a, F(d.O, d, "i", {xa: !1}), 10), {Nd: k, pg: B, stop: B};
                    Xa(Ta())
                }
                d.D(["sr"], function (r) {
                    var v, w = Qh(a, r.source);
                    w && zf(a, r.source,
                        (v = {}, v.type = "\u043d", v.frameId = c.va().$(w), v))
                });
                d.D(["sd"], function (r) {
                    var v = r.data;
                    r = r.source;
                    (a === r || Qh(a, r)) && d.O("sdr", {data: v.data, M: v.frameId})
                });
                if (f && !g) {
                    var l = !1, m = 0, p = function () {
                        var r;
                        zf(a, a.parent, (r = {}, r.type = "sr", r));
                        m = V(a, p, 100, "if.i")
                    };
                    p();
                    var q = function (r) {
                        d.oa(["\u043d"], q);
                        na(a, m);
                        var v = Hc(a, r.origin).host;
                        l || r.source !== a.parent || !r.data.frameId || "about:blank" !== S(a).host && !K(v, b) || (l = !0, d.O("i", {
                            M: r.data.frameId,
                            xa: !0
                        }))
                    };
                    d.D(["\u043d"], q);
                    V(a, function () {
                        d.oa(["\u043d"], q);
                        na(a,
                            m);
                        l || (l = !0, d.O("i", {xa: !1}))
                    }, 2E3, "if.r")
                }
                e = e.D(a, ["message"], function (r) {
                    var v = tb(a, r.data);
                    v && v.type && K(v.type, cn) && d.O(v.type, {data: v, source: r.source, origin: r.origin})
                });
                return {
                    Nd: k, pg: function (r) {
                        var v;
                        return zf(a, a.parent, (v = {}, v.frameId = c.Od(), v.data = r, v.type = "sd", v))
                    }, stop: e
                }
            }

            function Qh(a, c) {
                try {
                    return bb(t(X("contentWindow"), ka(c)), ya(a.document.querySelectorAll("iframe")))
                } catch (b) {
                    return null
                }
            }

            function zf(a, c, b) {
                a = mb(a, b);
                c.postMessage(a, "*")
            }

            function Ph() {
                return Zb() + Zb() + "-" + Zb() + "-" +
                    Zb() + "-" + Zb() + "-" + Zb() + Zb() + Zb()
            }

            function Zb() {
                return Math.floor(65536 * (1 + Math.random())).toString(16).substring(1)
            }

            function dn(a, c) {
                if (wa(c)) return c;
                var b = a.textContent;
                if (wa(b)) return b;
                b = a.data;
                if (wa(b)) return b;
                b = a.nodeValue;
                return wa(b) ? b : ""
            }

            function en(a, c, b, d, e) {
                void 0 === d && (d = {});
                void 0 === e && (e = Ia(c));
                var f = y(M(function (h, k) {
                    h[k.name] = k.value;
                    return h
                }, {}, ya(c.attributes)), d);
                y(f, fn(c, e, f));
                var g = (d = M(function (h, k) {
                    var l = k[0], m = ge(a, c, l, k[1], b, e), p = m.value;
                    ma(p) ? delete f[l] : f[l] = p;
                    return h ||
                        m.vb
                }, !1, pa(f))) && Ic(c);
                g && (f.width = g.width, f.height = g.height);
                return {vb: d, ih: f}
            }

            function fn(a, c, b) {
                var d = {};
                Af(a) ? d.value = a.value || b.value : "IMG" !== c || b.src || (d.src = "");
                return d
            }

            function ge(a, c, b, d, e, f) {
                void 0 === f && (f = Ia(c));
                var g = {vb: !1, value: d};
                if (Af(c)) "value" === b ? !ma(d) && "" !== d && (b = e.ae, f = e.dg, e = jd(a, c), f ? (b = Jc(a, c, b), a = b.wb, c = b.pb, b = b.cb, g.vb = !c && (e || a)) : (g.vb = e, b = !(c && $b("ym-record-keys", c))), b || e) && (g.value = J("", A(u("\u2022", O), ("" + d).split("")))) : "checked" === b && K((c.getAttribute("type") ||
                    "").toLowerCase(), gn) ? g.value = c.checked ? "checked" : null : hn.test(b) && Bf(a, c) && (g.value = null); else if ("IMG" === f && "src" === b) (e = jd(a, c)) ? (g.vb = e, g.value = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=") : g.value = (c.getAttribute("srcset") ? c.currentSrc : "") || c.src; else if ("A" === f && "href" === b) g.value = d ? "#" : ""; else if (K(b, ["srcset", "integrity", "crossorigin", "password"]) || 2 < b.length && 0 === he(b, "on") || "IFRAME" === f && "src" === b || "SCRIPT" === f && K(b,
                    ["src", "type"])) g.value = null;
                return g
            }

            function Cf(a, c, b, d) {
                void 0 === d && (d = "wv2");
                return {
                    J: function (e, f) {
                        return D(a, d + "." + b + "." + f, e, void 0, c)
                    }
                }
            }

            function jn(a, c, b, d, e) {
                function f() {
                    k && k.stop()
                }

                if (!c.Kb) return I.resolve(B);
                var g = Ba(a, "4", c), h = {H: Da()};
                b = new kn(a, b, function (l, m, p) {
                    var q;
                    if (!g) return I.resolve();
                    m = "wv-data=" + Rh(l, !0);
                    return g(y({}, h, {
                        Y: {fa: m},
                        G: (q = {}, q["wv-check"] = "" + ln(l), q["wv-type"] = "0", q)
                    }), c, p)["catch"](D(a, "m.n.m.s"))
                });
                var k = mn(a, b, d, e);
                return ra(c, function (l) {
                    l && H(a).C("isEU",
                        n(l, "settings.eu"));
                    H(a).o("oo") || k && Sh(a, l) && k.start();
                    return f
                })
            }

            function mn(a, c, b, d) {
                var e = a.document, f = [], g = ia(a), h = ":submit" + Math.random(), k = [], l = F(c.flush, c),
                    m = la(function (r, v) {
                        D(a, "hfv." + r, function () {
                            try {
                                var w = v.type
                            } catch (G) {
                                return
                            }
                            w = K(w, d);
                            c.push(v, {type: r});
                            w && l()
                        })()
                    }), p = D(a, "sfv", function () {
                        var r = b(a), v = nn(a);
                        z(function (w) {
                            f.push(g.D(w.target, [w.event], m(w.type)))
                        }, r);
                        z(function (w) {
                                f.push(g.D(w.target, [w.event], D(a, "hff." + w.type + "." + w.event, function (G) {
                                    z(za({l: a, sa: G, flush: l}), w.N)
                                })))
                            },
                            v);
                        k = Th(a, "form", e);
                        e.attachEvent && (r = Th(a, "form *", e), z(function (w) {
                            f.push(g.D(w, ["submit"], m("form")))
                        }, k), z(function (w) {
                            Df(w) && f.push(g.D(w, ["change"], m("formInput")))
                        }, r));
                        z(function (w) {
                            var G = w.submit;
                            if (T(G) || "object" === typeof G && on.test("" + G)) w[h] = G, w.submit = D(a, "fv", function () {
                                var Y = {target: w, type: "submit"};
                                m("document")(Y);
                                return w[h]()
                            })
                        }, k)
                    }), q = D(a, "ufv", function () {
                        z(ha, f);
                        z(function (r) {
                            r && (r.submit = r[h])
                        }, k);
                        c.flush()
                    });
                return {start: p, stop: q}
            }

            function pn(a, c) {
                var b = Z(function (e) {
                    return 0 <
                        e.N.length
                }, c), d = Uh({target: a.document, type: "document"});
                return A(t(O, d, qn(a)), b)
            }

            function Vh(a, c) {
                var b = a.l, d = [], e = c.form;
                if (!c[Wa] && e) {
                    var f = e.elements;
                    e = e.length;
                    for (var g = 0; g < e; g += 1) {
                        var h = f[g];
                        ie(h) && !h[Wa] && Oa(d, rc(b, h))
                    }
                } else Oa(d, rc(b, c));
                return d
            }

            function Ef(a) {
                if (kd) {
                    kd = !1;
                    var c = ub(a.l), b = [];
                    fb(a.l, b, 15) ? a = [] : (R(b, c), a = b);
                    return a
                }
            }

            function Wh(a) {
                if (!kd) {
                    kd = !0;
                    a = ub(a.l);
                    var c = [];
                    Mb(c, 14);
                    R(c, a);
                    return c
                }
            }

            function rn(a, c, b) {
                var d = c[Wa];
                if (d) {
                    a:{
                        var e = ub(a), f = c[Wa];
                        if (0 < f) {
                            var g = [];
                            c = Ff(a,
                                c);
                            var h = sc[f], k = c[0] + "x" + c[1], l = c[2] + "x" + c[3];
                            if (k !== h.$f) {
                                h.$f = k;
                                if (fb(a, g, 9)) {
                                    a = [];
                                    break a
                                }
                                R(g, e);
                                R(g, f);
                                R(g, c[0]);
                                R(g, c[1])
                            }
                            if (l !== h.size) {
                                h.size = l;
                                if (fb(a, g, 10)) {
                                    a = [];
                                    break a
                                }
                                R(g, e);
                                R(g, f);
                                R(g, c[2]);
                                R(g, c[3])
                            }
                            if (g.length) {
                                a = g;
                                break a
                            }
                        }
                        a = []
                    }
                    Oa(b, a)
                }
                return d
            }

            function Jc(a, c, b) {
                void 0 === b && (b = !1);
                if (!c) return {cb: !1, pb: !1, wb: !1};
                var d = c.getAttribute("type") || c.type;
                if ("button" === d) return {cb: !1, pb: !1, wb: !1};
                var e = Z(Xh, [c.className, c.id, c.name]), f = c && $b("ym-record-keys", c);
                d = d && K(d, Yh) || Ja(Ya(sn),
                    e);
                var g;
                (g = d) || (g = c.placeholder, g = Ja(Ya(tn), e) || Xh(g) && un.test(g || ""));
                e = g;
                return {cb: !f && (Gf(a, c) || e && b || e && !d && !b), pb: f, wb: e}
            }

            function Gf(a, c) {
                return Bf(a, c) || ld(a, c) ? !0 : jd(a, c)
            }

            function Xh(a) {
                return !!(a && 2 < a.length)
            }

            function Af(a) {
                try {
                    var c = Ia(a);
                    if (K(c, Hf)) {
                        if ("INPUT" === c) {
                            var b = a.type;
                            return !b || K(b.toLocaleLowerCase(), vn)
                        }
                        return !0
                    }
                } catch (d) {
                }
                return !1
            }

            function Zh(a, c) {
                return c && $b("(ym-disable-submit|-metrika-noform)", c)
            }

            function wn(a, c) {
                return J("", A(function (b) {
                    return a.isNaN(b) ? xn.test(b) ?
                        (b = b.toUpperCase() === b ? yn : zn, String.fromCharCode(Va(a, b[0], b[1]))) : b : "" + Va(a, 0, 9)
                }, c.split("")))
            }

            function jd(a, c) {
                if (ma(c)) return !1;
                if (If(c)) {
                    var b = c.parentNode;
                    return (ma(b) ? 0 : 11 === b.nodeType) ? !1 : jd(a, c.parentNode)
                }
                b = $h(a);
                if (!b) return !1;
                var d = b.call(c, ".ym-hide-content,.ym-hide-content *");
                return d && b.call(c, ".ym-show-content,.ym-hide-content .ym-show-content *") ? !1 : d
            }

            function Sh(a, c) {
                var b = ac(a), d = b.o("visorc");
                K(d, ["w", "b"]) || (d = "");
                ai(a) && bi(a, je, "visorc") && !An.test(gb(a) || "") || (d = "b");
                var e =
                    n(c, "settings.webvisor.recp");
                if (!a.isFinite(e) || 0 > e || 1 < e) d = "w";
                d || (d = H(a).o("hitId") % 1E4 / 1E4 < e ? "w" : "b");
                b.C("visorc", d, 30);
                return "w" === d
            }

            function ci(a) {
                var c = Jf(a).isEnabled, b = !1;
                try {
                    b = (b = 2 === (new a.Blob(["\u00e4"])).size) && 2 === (new a.Blob([new a.Uint8Array([1, 2])])).size
                } catch (d) {
                }
                return Kf(Boolean, [!c, b, a.Uint8Array, n(a, "Uint8Array.prototype.slice")])
            }

            function Lf(a, c) {
                var b = c[1][3], d = 0, e = new a.Uint8Array(c[0]);
                return jc([b], function (f, g) {
                    if (!f) return e;
                    f[0](a, f[2], e, d);
                    d += f[1];
                    g.push(f[3]);
                    return e
                })
            }

            function ke(a, c, b) {
                a = c(b);
                c = [B, 0, 0];
                var d = [0, c, c, void 0];
                return jc(a, function (e, f) {
                    var g = e[0], h = e[1], k = e[2];
                    if (0 === g) return k(d, h), d;
                    if (void 0 === h || null === h) return d;
                    var l = g >> 3;
                    if (g & 1) tc(d, nb(l)), h = k(h), l & 2 && tc(d, nb(h[1])), tc(d, h); else if (g & 4) for (g = h.length - 1; 0 <= g;) {
                        var m = k(h[g]);
                        m.push([0, 0, Mf]);
                        m.push([0, nb(l), tc]);
                        m.unshift([0, 0, Nf]);
                        f.push.apply(f, m);
                        --g
                    } else if (g & 2) {
                        k = e[2];
                        var p = e[3], q = e[4], r = e[5], v = da(h);
                        for (g = v.length - 1; 0 <= g;) m = v[g], m = [[0, 0, Nf], [q, h[m], r], [k, m, p], [0, 0, Mf], [0, nb(l), tc]], f.push.apply(f,
                            m), --g
                    } else m = k(h), m.push([0, 0, Mf]), m.push([0, nb(l), tc]), m.unshift([0, 0, Nf]), f.push.apply(f, m);
                    return d
                })
            }

            function Nf(a) {
                var c = a[1], b = a[0], d = a[2];
                a[3] ? (a[0] = a[3][0], a[1] = a[3][1], a[2] = a[3][2], a[3] = a[3][3]) : (a[0] = 0, a[1] = [B, 0, 0], a[2] = a[1]);
                tc(a, nb(b));
                b && (a[2][3] = c[3], a[2] = d, a[0] += b)
            }

            function Mf(a) {
                a[3] = [a[0], a[1], a[2], a[3]];
                a[1] = [B, 0, 0];
                a[2] = a[1];
                a[0] = 0
            }

            function tc(a, c) {
                a[0] += c[1];
                a[2][3] = c;
                a[2] = c
            }

            function Of(a) {
                return [[385, a.eh, nb], [336, a.qj, Bn], [272, a.gh, Cn], [208, a.event, Dn], [144, a.Vi, En], [80, a.page,
                    Fn]]
            }

            function Gn(a) {
                return [[321, a.end, Nb], [273, a.uh, Hn], [193, a.page, L], [144, a.data, Of], [65, a.L, L]]
            }

            function di(a) {
                return [[84, a.buffer, Gn]]
            }

            function In(a) {
                return [[129, a.position, L], [81, a.name, P]]
            }

            function Jn(a) {
                return [[81, a.name, P]]
            }

            function Kn(a) {
                return [[81, a.name, P]]
            }

            function Cn(a) {
                return [[593, a.Uj, P], [532, a.vj, In], [449, a.kf, L], [401, a.pj, P], [340, a.Jj, Jn], [276, a.jh, Kn], [209, a.lj, P], [145, a.mj, P], [65, a.id, nb]]
            }

            function Ln(a) {
                return [[513, a.kf, L], [489, a.Si, md], [385, a.Ra, L], [321, a.height, L], [257, a.width,
                    L], [193, a.y, L], [129, a.x, L], [65, a.id, nb]]
            }

            function Bn(a) {
                return [[129, a.Ra, L], [84, a.hh, Ln]]
            }

            function Mn(a) {
                return [[81, a.hash, P]]
            }

            function Nn(a) {
                return [[209, a.stack, P], [145, a.Jh, P], [81, a.code, P]]
            }

            function On(a) {
                return [[193, a.orientation, L], [129, a.height, L], [65, a.width, L]]
            }

            function Pn(a) {
                return [[84, a.Sc, Qn]]
            }

            function Qn(a) {
                return [[273, a.Tc, P], [193, a.Of, Nb], [145, a.key, P], [65, a.id, L]]
            }

            function Rn(a) {
                return [[145, a.value, P], [81, a.Ik, P]]
            }

            function Sn(a) {
                return [[149, a.Nb, P], [81, a.method, P]]
            }

            function Tn(a) {
                return [[257,
                    a.Ab, L], [193, a.Bb, L], [129, a.height, L], [65, a.width, L]]
            }

            function Un(a) {
                return [[144, a.Sg, ei], [80, a.Qg, ei]]
            }

            function ei(a) {
                return [[193, a.y, L], [129, a.x, L], [105, a.level, md]]
            }

            function Vn(a) {
                return [[84, a.touches, Wn]]
            }

            function Wn(a) {
                return [[297, a.force, md], [233, a.y, md], [169, a.x, md], [81, a.id, P]]
            }

            function Xn(a) {
                return [[193, a.hidden, Nb], [129, a.checked, Nb], [81, a.value, P]]
            }

            function Yn(a) {
                return [[257, a.pf, L], [193, a.yg, L], [129, a.end, L], [65, a.start, L]]
            }

            function Zn() {
                return []
            }

            function $n(a) {
                return [[193, a.page, Nb],
                    [129, a.y, L], [65, a.x, L]]
            }

            function ao(a) {
                return [[129, a.y, L], [65, a.x, L]]
            }

            function bo(a) {
                return [[84, a.Oa, co]]
            }

            function co(a) {
                return [[257, a.index, L], [209, a.te, P], [145, a.style, P], [65, a.target, L]]
            }

            function Dn(a) {
                return [[1168, a.Cj, bo], [1104, a.Ai, Mn], [1040, a.Mh, Nn], [976, a.Ch, On], [912, a.Pi, Pn], [848, a.sj, Tn], [784, a.ck, Un], [720, a.Jk, Rn], [656, a.Fk, Sn], [592, a.Kj, Vn], [528, a.ph, Xn], [464, a.yj, Yn], [400, a.$j, Zn], [336, a.wj, $n], [272, a.Ti, ao], [193, a.M, le], [129, a.target, le], [65, a.type, nb]]
            }

            function En(a) {
                return [[257, a.M,
                    le], [208, a.aa, eo], [129, a.L, L], [65, a.target, L]]
            }

            function eo(a) {
                return [[148, a.Oa, fo], [65, a.index, L]]
            }

            function fo(a) {
                return [[276, a.d, go], [212, a.wd, ho], [148, a.gf, io], [84, a.cf, jo]]
            }

            function go(a) {
                return [[193, a.Ha, L], [144, a.Ac, fi], [65, a.id, L]]
            }

            function ho(a) {
                return [[193, a.Ha, L], [146, a.Ob, 81, P, 144, fi], [65, a.id, L]]
            }

            function fi(a) {
                return [[193, a.r, Nb], [145, a.n, P], [81, a.Wc, P]]
            }

            function io(a) {
                return [[641, a.Qa, Nb], [577, a.Ha, L], [513, a.ne, L], [465, a.Ac, P], [402, a.Ob, 81, P, 145, P], [321, a.xe, L], [273, a.Xf, P], [193, a.ue,
                    L], [145, a.Wf, P], [65, a.id, L]]
            }

            function jo(a) {
                return [[321, a.Ha, L], [257, a.ue, L], [193, a.ne, L], [129, a.xe, L], [65, a.id, L]]
            }

            function Fn(a) {
                return [[321, a.Ka, ko], [273, a.ld, P], [193, a.M, le], [148, a.content, lo], [80, a.aa, mo]]
            }

            function lo(a) {
                return [[513, a.hidden, Nb], [449, a.ze, L], [385, a.next, L], [337, a.content, P], [257, a.parent, L], [210, a.attributes, 81, P, 145, P], [145, a.name, P], [65, a.id, L]]
            }

            function mo(a) {
                return [[724, a.Bk, no], [656, a.location, oo], [592, a.viewport, gi], [528, a.screen, gi], [449, a.Jf, Nb], [401, a.hf, P], [337, a.referrer,
                    P], [273, a.Jg, P], [209, a.ef, P], [145, a.title, P], [81, a.doctype, P]]
            }

            function no(a) {
                return [[133, a.scroll, L], [65, a.target, L]]
            }

            function oo(a) {
                return [[209, a.path, P], [145, a.protocol, P], [81, a.host, P]]
            }

            function gi(a) {
                return [[129, a.height, L], [65, a.width, L]]
            }

            function P(a) {
                var c = po({}, a, [], 0);
                return c ? [qo, c, a] : [hi, 0, 0]
            }

            function Hn(a) {
                return [ro, a.length, a]
            }

            function Nb(a) {
                return [hi, 1, a ? 1 : 0]
            }

            function ko(a) {
                a = ii(a);
                var c = a[0], b = a[1], d = (b >>> 28 | c << 4) >>> 0;
                c >>>= 24;
                return [ji, 0 === c ? 0 === d ? 16384 > b ? 128 > b ? 1 : 2 : 2097152 > b ? 3 : 4 : 16384 >
                d ? 128 > d ? 5 : 6 : 2097152 > d ? 7 : 8 : 128 > c ? 9 : 10, a]
            }

            function md(a) {
                return [so, 4, a]
            }

            function le(a) {
                return nb((a << 1 ^ a >> 31) >>> 0)
            }

            function L(a) {
                return 0 > a ? [ji, 10, ii(a)] : nb(a)
            }

            function nb(a) {
                return [to, 128 > a ? 1 : 16384 > a ? 2 : 2097152 > a ? 3 : 268435456 > a ? 4 : 5, a]
            }

            function to(a, c, b, d) {
                for (a = c; 127 < a;) b[d++] = a & 127 | 128, a >>>= 7;
                b[d] = a
            }

            function hi(a, c, b, d) {
                b[d] = c
            }

            function ro(a, c, b, d) {
                for (a = 0; a < c.length; ++a) b[d + a] = c[a]
            }

            function ki(a) {
                return function (c, b, d, e) {
                    for (var f, g = 0, h = 0; h < b.length; ++h) if (c = b.charCodeAt(h), 128 > c) a ? g += 1 : d[e++] = c; else {
                        if (2048 >
                            c) {
                            if (a) {
                                g += 2;
                                continue
                            }
                            d[e++] = c >> 6 | 192
                        } else {
                            if (55296 === (c & 64512) && 56320 === ((f = b.charCodeAt(h + 1)) & 64512)) {
                                if (a) {
                                    g += 4;
                                    continue
                                }
                                c = 65536 + ((c & 1023) << 10) + (f & 1023);
                                ++h;
                                d[e++] = c >> 18 | 240;
                                d[e++] = c >> 12 & 63 | 128
                            } else {
                                if (a) {
                                    g += 3;
                                    continue
                                }
                                d[e++] = c >> 12 | 224
                            }
                            d[e++] = c >> 6 & 63 | 128
                        }
                        d[e++] = c & 63 | 128
                    }
                    return a ? g : e
                }
            }

            function so(a, c, b, d) {
                return uo(a)(a, c, b, d)
            }

            function vo(a, c, b, d) {
                var e = 0 > c ? 1 : 0;
                e && (c = -c);
                if (0 === c) nd(0 < 1 / c ? 0 : 2147483648, b, d); else if (a.isNaN(c)) nd(2143289344, b, d); else if (3.4028234663852886E38 < c) nd((e << 31 | 2139095040) >>>
                    0, b, d); else if (1.1754943508222875E-38 > c) nd((e << 31 | a.Math.round(c / 1.401298464324817E-45)) >>> 0, b, d); else {
                    var f = a.Math.floor(a.Math.log(c) / Math.LN2);
                    nd((e << 31 | f + 127 << 23 | Math.round(c * a.Math.pow(2, -f) * 8388608) & 8388607) >>> 0, b, d)
                }
            }

            function nd(a, c, b) {
                c[b] = a & 255;
                c[b + 1] = a >>> 8 & 255;
                c[b + 2] = a >>> 16 & 255;
                c[b + 3] = a >>> 24
            }

            function ji(a, c, b, d) {
                a = c[0];
                for (c = c[1]; a;) b[d++] = c & 127 | 128, c = (c >>> 7 | a << 25) >>> 0, a >>>= 7;
                for (; 127 < c;) b[d++] = c & 127 | 128, c >>>= 7;
                b[d++] = c
            }

            function ii(a) {
                if (!a) return [0, 0];
                var c = 0 > a;
                c && (a = -a);
                var b = a >>> 0;
                a =
                    (a - b) / 4294967296 >>> 0;
                c && (a = ~a >>> 0, b = ~b >>> 0, 4294967295 < ++b && (b = 0, 4294967295 < ++a && (a = 0)));
                return [a, b]
            }

            function me(a) {
                return ca(a) ? A(me, a) : ma(a) || "object" !== typeof a ? a : M(function (c, b) {
                    var d = b[0], e = b[1], f = wo[d];
                    W(f) && (f = d);
                    e = K(f, xo) ? e : me(e);
                    f ? c[f] = e : c[d] = e;
                    return c
                }, {}, pa(a))
            }

            function li(a, c, b) {
                return function (d, e, f) {
                    var g;
                    d = y({H: Da()}, d);
                    d.H.fc("we", Fb(e.Kb));
                    d.G || (d.G = {});
                    var h = d.G, k = d.Ta;
                    k = void 0 === k ? {} : k;
                    h.wmode = "0";
                    h["wv-part"] = "" + f;
                    h["wv-hit"] = h["wv-hit"] || "" + Lb(a);
                    h["page-url"] = h["page-url"] ||
                        a.location.href;
                    h.rn = h.rn || "" + Va(a);
                    h["wv-type"] || (h["wv-type"] = k.Zd ? "3" : "2");
                    f = {
                        ja: {ta: "webvisor/" + e.id},
                        Y: y(d.Y, {Cb: (g = {}, g["Content-Type"] = "text/plain", g), Ze: "POST"}),
                        G: h
                    };
                    e = Oa(Pf(a, "wv", e), b);
                    return ne(a, c, e)(y(d, f))
                }
            }

            function yo(a, c) {
                return ra(c, function (b) {
                    var d = H(a);
                    N(c);
                    if (!d.o("dSync", !1)) return d.C("dSync", !0), mi(a, b, {
                        kb: c, bd: "s", ie: "ds", Ej: function (e, f, g) {
                            var h = e.fd;
                            e = e.host;
                            if (n(h, "settings")) return Xa(Ta("ds.e"));
                            f = f(aa) - g;
                            g = e[1];
                            var k, l;
                            h = Da((k = {}, k.di = h, k.dit = f, k.dip = g, k));
                            k = (l = {},
                                l["page-url"] = S(a).href, l);
                            return Ba(a, "S", ni)({H: h, G: k}, ni)["catch"](D(a, "ds.rs"))
                        }
                    })
                })
            }

            function mi(a, c, b) {
                var d = b.kb, e = fa(a), f = zo(a, c.userData, d), g = Ao(a);
                return g.length ? Bo(a, e, f, c, b).then(function () {
                    return Co(a, g, f, e, b)
                }, B) : I.resolve()
            }

            function Ao(a) {
                var c = od(a);
                a = t(Qf, uc(["iPhone", "iPad"]))(a);
                return c ? Do : a ? Eo : []
            }

            function Co(a, c, b, d, e) {
                var f = e.Ej, g = void 0 === f ? B : f, h = e.ie, k = d(aa);
                return Fo(a, c, e)(Sa(function (l) {
                    z(function (m) {
                        m && oe(a, h + ".s", m)
                    }, l);
                    l = d(lb);
                    b.C(h, l)
                }, function (l) {
                    b.C(h, d(lb));
                    g(l, d,
                        k)
                }))
            }

            function Bo(a, c, b, d, e) {
                var f = e.ie, g = e.kb;
                return new I(function (h, k) {
                    var l = b.o(f, 0);
                    l = parseInt("" + l, 10);
                    return 60 >= c(lb) - l ? k() : Go(a) ? h(void 0) : oi(d) ? k() : h(Ho(a, g))
                })
            }

            function Fo(a, c, b) {
                var d = b.bd, e = b.data, f = Ba(a, d, b.kb);
                a = y({}, pi);
                e && y(a.G, e);
                return Io(A(function (g) {
                    return Jo(f(y({Y: {Pg: !1, uj: !0}}, pi), A(function (h) {
                        var k = h[1], l = h[2];
                        h = J("", A(function (m) {
                            return String.fromCharCode(m.charCodeAt(0) + 10)
                        }, h[0].split("")));
                        return "http" + (l ? "s" : "") + "://" + h + ":" + k + "/" + Ko[d]
                    }, g)).then(function (h) {
                        return y({},
                            h, {host: g[h.Lg]})
                    }))
                }, c))
            }

            function zo(a, c, b) {
                var d = c || {}, e = Ba(a, "u", b), f = Ra(a);
                return {
                    o: function (g, h) {
                        return W(d[g]) ? f.o(g, h) : d[g]
                    }, C: function (g, h) {
                        var k, l = "" + h;
                        d[g] = l;
                        f.C(g, l);
                        return e({G: (k = {}, k.key = g, k.value = l, k)}, [sa.Za + "//" + bc + "/user_storage_set"], {})["catch"](D(a, "u.d.s.s"))
                    }
                }
            }

            function Lo(a) {
                return {
                    Z: function (c, b) {
                        H(a).o("oo") || b()
                    }
                }
            }

            function Mo(a, c) {
                try {
                    var b = c[0];
                    var d = b[1]
                } catch (e) {
                    return function () {
                        return I.resolve()
                    }
                }
                return function (e) {
                    var f, g = (f = {}, f["browser-info"] = No, f["page-url"] = a.location &&
                        "" + a.location.href, f);
                    return d && (e = mb(a, e)) ? d(Oo, {
                        bc: g,
                        Fa: [],
                        fa: "site-info=" + fe(e)
                    })["catch"](B) : I.resolve()
                }
            }

            function Po(a, c) {
                if (n(a, "disableYaCounter" + c.id) || n(a, "Ya.disableMetrica")) {
                    var b = N(c);
                    delete H(a).o("counters", {})[b];
                    Xa(Ta("oo.e"))
                }
            }

            function Qo(a) {
                if (pd(a)) return null;
                var c = Ro(a), b = c.cg;
                W(b) && (c.cg = null, So(a).then(function (d) {
                    c.cg = d
                }));
                return b ? 1 : null
            }

            function To(a, c, b) {
                b = b.G;
                if ((void 0 === b ? {} : b).nohit) return null;
                a = Rf(a);
                if (!a) return null;
                var d = b = null;
                n(a, "getEntriesByType") && (d = n(a.getEntriesByType("navigation"),
                    "0")) && (b = Uo);
                if (!b) {
                    var e = n(a, "timing");
                    e && (b = Vo, d = e)
                }
                if (!b) return null;
                a = Wo(a, d, b);
                c = N(c);
                c = Xo(c);
                return (c = Yo(c, a)) && J(",", c)
            }

            function Yo(a, c) {
                var b = a.length ? A(function (d, e) {
                    var f = c[e];
                    return f === d ? null : f
                }, a) : c;
                a.length = 0;
                z(t(O, qa("push", a)), c);
                return Z(ka(null), b).length === a.length ? null : b
            }

            function Wo(a, c, b) {
                return A(function (d) {
                    var e = d[0], f = d[1];
                    if (T(e)) return e(a, c) || null;
                    if (1 === d.length) return c[e] ? Math.round(c[e]) : null;
                    var g;
                    !(g = c[e] && c[f]) && (g = 0 === c[e] && 0 === c[f]) && (g = d[1], g = !(qi[d[0]] || qi[g]));
                    if (!g) return null;
                    d = Math.round(c[e]) - Math.round(c[f]);
                    return 0 > d || 36E5 < d ? null : d
                }, b)
            }

            function Rh(a, c) {
                void 0 === c && (c = !1);
                for (var b = a.length, d = b - b % 3, e = [], f = 0; f < d; f += 3) {
                    var g = (a[f] << 16) + (a[f + 1] << 8) + a[f + 2];
                    e.push("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="[g >> 18 & 63], "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="[g >> 12 & 63], "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="[g >> 6 & 63], "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="[g &
                    63])
                }
                switch (b - d) {
                    case 1:
                        b = a[d] << 4;
                        e.push("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="[b >> 6 & 63], "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="[b & 63], "=", "=");
                        break;
                    case 2:
                        b = (a[d] << 10) + (a[d + 1] << 2), e.push("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="[b >> 12 & 63], "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="[b >> 6 & 63], "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="[b & 63], "=")
                }
                e = e.join("");
                return c ?
                    ri(e, !0) : e
            }

            function Ah(a, c) {
                void 0 === c && (c = !1);
                var b = a, d = "", e = 0;
                if (!b) return "";
                for (c && (b = ri(b)); b.length % 4;) b += "=";
                do {
                    var f = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".indexOf(b.charAt(e++)),
                        g = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".indexOf(b.charAt(e++)),
                        h = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".indexOf(b.charAt(e++)),
                        k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".indexOf(b.charAt(e++));
                    if (0 > f ||
                        0 > g || 0 > h || 0 > k) return "";
                    var l = f << 18 | g << 12 | h << 6 | k;
                    f = l >> 16 & 255;
                    g = l >> 8 & 255;
                    l &= 255;
                    d = 64 === h ? d + String.fromCharCode(f) : 64 === k ? d + String.fromCharCode(f, g) : d + String.fromCharCode(f, g, l)
                } while (e < b.length);
                return d
            }

            function ri(a, c) {
                void 0 === c && (c = !1);
                return a ? a.replace(c ? /[+/=]/g : /[-*_]/g, function (b) {
                    return Zo[b] || b
                }) : ""
            }

            function $o(a) {
                try {
                    var c = Pa(a) ? a : [];
                    return J(",", [a.name, a.description, t(ya, Na, hb(ap), qd(","))(c)])
                } catch (b) {
                    return ""
                }
            }

            function ap(a) {
                return J(",", [a.description, a.suffixes, a.type])
            }

            function bp(a,
                        c) {
                for (var b = "", d = 0; d < c; d += 1) b += a;
                return b
            }

            function cp(a, c, b, d, e, f, g, h) {
                var k = b.o(f);
                ma(k) && (b.C(f, g), e(a, c, b, d), k = b.o(f, g));
                W(h) || h.fc(f, "" + k);
                return k
            }

            function dp(a, c) {
                if (rd(a)) {
                    var b = gb(a).match(ep);
                    if (b && b.length) return b[1] === c
                }
                return !1
            }

            function pe(a, c, b) {
                return function (d) {
                    var e, f, g = Ha(c, b);
                    g && fp(a, d, c) && (g = F(g.params, g), (d = Sf({
                        event: a,
                        Ma: "products",
                        za: vc,
                        si: "goods"
                    }, d)) && g && g((e = {}, e.__ym = (f = {}, f.ecommerce = [d], f), e)))
                }
            }

            function fp(a, c, b) {
                var d = !1, e = "";
                if (!La(c)) return Db(b, "", "Ecommerce data should be an object"),
                    d;
                var f = c.goods;
                switch (a) {
                    case "detail":
                    case "add":
                    case "remove":
                        ca(f) && f.length ? (d = Kf(function (g) {
                            return La(g) && (wa(g.id) || qe(b, g.id) || wa(g.name))
                        }, f)) || (e = "All items in 'goods' should be objects and contain 'id' or 'name' field") : e = "Ecommerce data should contain 'goods' non-empty array";
                        break;
                    case "purchase":
                        qe(b, c.id) || wa(c.id) ? d = !0 : e = "Purchase object should contain string or number 'id' field"
                }
                Db(b, "", e);
                return d
            }

            function sd(a, c) {
                return {
                    Z: function (b, d) {
                        re(b) ? d() : ra(c, function (e) {
                            var f;
                            if (e = n(e,
                                "settings.hittoken")) e = (f = {}, f.hittoken = e, f), b.G = y(b.G || {}, e);
                            d()
                        })
                    }
                }
            }

            function gp(a, c) {
                function b() {
                    q.hidden ? y(k.style, td(["top", "right", "left", "background"], "initial")) : y(k.style, td(["top", "right", "left"], "0"), {background: "rgba(0, 0, 0, .3)"});
                    w.parentNode || (r.appendChild(p), r.appendChild(w));
                    q.hidden = !q.hidden;
                    r.hidden = !r.hidden;
                    v.hidden = !v.hidden
                }

                function d(Q) {
                    var oa = g();
                    y(oa.style, wc("2px", "18px"), Kc, {
                        left: "15px",
                        top: "7px",
                        background: "#2f3747",
                        borderRadius: "2px"
                    });
                    oa.style.transform = "rotate(" +
                        Q + "deg)";
                    return oa
                }

                function e(Q, oa, ta, vb, ud) {
                    var se = g();
                    y(se.style, wc(oa + "px", ta + "px"), Kc, {
                        left: Q + "px",
                        bottom: 0,
                        background: vb,
                        borderTopLeftRadius: ud
                    });
                    return se
                }

                var f = ab(a);
                if (!f) return B;
                var g = u("div", f), h = u("iframe", f), k = g();
                k.classList.add("__ym_wv_ign");
                y(k.style, si, {bottom: "0", width: "100%", maxWidth: "initial", zIndex: "999999999"});
                var l = k.attachShadow ? k.attachShadow({mode: "open"}) : k, m = g();
                y(m.style, wc("24px"), Kc, Tf, {top: "12px", right: "10px", background: "#3367dc", overflow: "hidden"});
                var p = g();
                y(p.style,
                    {
                        border: "2px solid transparent",
                        animation: "__ym_wv_ign-spinner-animation 1s 0.21s infinite linear"
                    }, Tf, Kc, wc("48px"), td(["top", "left"], "calc(50% - 24px)"), td(["borderTopColor", "borderLeftColor"], "#fc0"));
                f = f("style");
                f.textContent = "@keyframes __ym_wv_ign-spinner-animation {to {transform: rotate(360deg);}}";
                p.appendChild(f);
                var q = g();
                q.id = "__ym_wv_ign__opener";
                y(q.style, wc("46px", "48px"), si, {
                    right: "0",
                    bottom: "60px",
                    cursor: "pointer",
                    background: "#fff",
                    borderRadius: "16px 0 0 16px",
                    boxShadow: "0px 0px 1px rgba(67, 68, 69, 0.3), 0px 1px 2px rgba(67, 68, 69, 0.3)"
                });
                var r = g();
                y(r.style, Kc, td(["top", "right", "bottom"], "0"), {width: "600px", background: "#fff"});
                var v = g();
                v.id = "__ym_wv_ign__closer";
                y(v.style, wc("32px"), Kc, Tf, {top: "12px", right: "612px", cursor: "pointer", background: "#fff"});
                f = h();
                f.src = "https://metrika.yandex.ru/widget/iframe-check";
                var w = h();
                y(w.style, wc("100%"), {border: "none"});
                w.src = "https://metrika.yandex.ru/widget/dashboard?id=" + c;
                r.hidden = !0;
                v.hidden = !0;
                v.appendChild(d(45));
                v.appendChild(d(-45));
                r.appendChild(f);
                m.appendChild(e(0, 8, 9, "linear-gradient(0deg, #ff324f, #ff324f), linear-gradient(158.67deg, #ff455c 12.6%, #ff1139 96.76%)"));
                m.appendChild(e(8, 9, 16, "#04acff", "3px"));
                m.appendChild(e(17, 7, 24, "#ffdd13"));
                q.appendChild(m);
                l.appendChild(r);
                l.appendChild(v);
                var G = ["click", "touchstart"];
                h = ia(a);
                m = a.document.body;
                l = [h.D(q, G, b), h.D(v, G, b), h.D(f, ["load"], E([ha, [F(r.removeChild, r, f), F(l.appendChild, l, q)]], z)), h.D(w, ["load"], F(r.removeChild, r, p)), F(m.removeChild, m, k)];
                var Y = E([ha, l], z);
                l.push(h.D(a, ["securitypolicyviolation"], function (Q) {
                    (Q = n(Q, "blockedURI")) && 0 <= Q.indexOf("https://metrika.yandex.ru") && Y()
                }));
                m.appendChild(k);
                return Y
            }

            function td(a, c) {
                return M(function (b, d) {
                    b[d] = c;
                    return b
                }, {}, a)
            }

            function wc(a, c) {
                var b;
                return b = {}, b.width = a, b.height = c || a, b
            }

            function hp(a, c) {
                try {
                    var b = c.origin
                } catch (d) {
                }
                b && Ja(t(Ya, za(b)), [/^http:\/\/([\w\-.]+\.)?webvisor\.com\/?$/, /^https:\/\/([\w\-.]+\.)?metri[kc]a\.yandex\.(ru|ua|by|kz|com|com\.tr)\/?$/]) && (b = tb(a, c.data), "appendremote" === n(b, "action") && ip(a, c, b))
            }

            function ti(a, c, b, d) {
                var e, f, g, h;
                void 0 === b && (b = "");
                void 0 === d && (d = "");
                var k = H(a), l = {};
                l.getCachedTags = Uf;
                l.form = (e = {}, e.closest = u(a,
                    ui), e.select = jp, e.getData = u(a, vi), e);
                l.button = (f = {}, f.closest = u(a, Vf), f.select = Wf, f.getData = u(a, Xf), f);
                l.phone = (g = {}, g.hidePhones = E([a, null, [d]], wi), g);
                l.status = (h = {}, h.checkStatus = E([a, Ga(b)], kp), h);
                k.C("_u", l);
                c && lc(a, {src: c})
            }

            function xi(a) {
                var c = a.lang;
                c = void 0 === c ? "" : c;
                var b = a.appVersion;
                b = void 0 === b ? "" : b;
                var d = a.fileId;
                d = void 0 === d ? "" : d;
                a = a.beta;
                a = void 0 === a ? !1 : a;
                b = J(".", t(hb(t(O, Ga)), Na)(b.split(".")));
                if (!K(d, lp) || !K(c, ["ru", "en", "tr"])) return "";
                c = (a ? "https://s3.mds.yandex.net/internal-metrika-betas" :
                    "https://yastatic.net/s3/metrika") + (b ? "/" + b : "") + "/form-selector/" + (d + "_" + c + ".js");
                return yi(c) ? c : ""
            }

            function mp(a, c) {
                var b = ab(a);
                if (b) {
                    var d = b("div"), e = Yb(a);
                    if (e) {
                        d.innerHTML = '<iframe name="RemoteIframe" allowtransparency="true" style="position: absolute; left: -999px; top: -999px; width: 1px; height: 1px;"></iframe>';
                        var f = d.firstChild;
                        f.onload = function () {
                            var h = b("meta");
                            h.setAttribute("http-equiv", "Content-Security-Policy");
                            h.setAttribute("content", "script-src *");
                            f.contentWindow.document.head.appendChild(h);
                            lc(f.contentWindow, {src: c})
                        };
                        a._ym__remoteIframeEl = f;
                        e.appendChild(d);
                        d.removeChild(f);
                        var g = null;
                        d.attachShadow ? g = d.attachShadow({mode: "open"}) : d.createShadowRoot ? g = d.createShadowRoot() : d.webkitCreateShadowRoot && (g = d.webkitCreateShadowRoot());
                        g ? g.appendChild(f) : (e.appendChild(f), a._ym__remoteIframeContainer = f)
                    }
                }
            }

            function kp(a) {
                var c, b = zi(a);
                a = H(a).o("getCounters", vd)();
                a = A(X("id"), a);
                return c = {id: b}, c.counterFound = !!b && K(b, a), c
            }

            function wi(a, c, b) {
                var d;
                c = Ai(a, c, {Gg: np, Xi: (d = {}, d.href = !0, d)});
                b = Z(Boolean,
                    A(function (f) {
                        return "*" === f ? f : Ob(f)
                    }, b));
                var e = A(t(O, qa("concat", [""]), Bi("reverse"), ha), b);
                b = wd(a);
                d = Ci(a, b, 1E3);
                c = F(c.fg, c, e);
                d.D(c);
                op(a, b);
                Di(a, b);
                c()
            }

            function np(a, c, b) {
                var d = ab(a), e = b.Db, f = b.dd, g = e.parentNode, h = e.textContent;
                if ("text" === b.Ge && h && d && g) {
                    b = d("small");
                    Ei(b);
                    var k = h.split(""), l = Fi(h).length;
                    z(qa("appendChild", b), M(function (m, p) {
                        var q = m.na, r = m.Og, v = d("small");
                        v.innerHTML = p;
                        var w = pp.test(p);
                        Ei(v);
                        w && (v.style.opacity = "" + (l - r - 1) / l);
                        q.push(v);
                        return {na: q, Og: r + (w ? 1 : 0)}
                    }, {na: [], Og: 0}, k).na);
                    qp(a, c, b, f);
                    g.insertBefore(b, e);
                    e.textContent = "";
                    return !0
                }
                return !1
            }

            function qp(a, c, b, d) {
                function e() {
                    z(u(["style", "opacity", ""], Gc), ya(b.childNodes));
                    if (c) {
                        var k = Ha(a, c);
                        k && k.extLink("tel:" + d, {})
                    }
                    g();
                    h()
                }

                var f = ia(a), g = B, h = B;
                g = f.D(b, ["mouseenter"], function (k) {
                    if (k.target === b) {
                        var l = V(a, e, 200, "ph.h.e");
                        (h || B)();
                        h = f.D(b, ["mouseleave"], function (m) {
                            m.target === b && na(a, l)
                        })
                    }
                })
            }

            function Di(a, c) {
                cc(a)(Sa(B, function () {
                    var b, d = a.document.body, e = (b = {}, b.attributes = !0, b.childList = !0, b.subtree = !0, b);
                    Aa("MutationObserver",
                        a.MutationObserver) && (new MutationObserver(c.O)).observe(d, e)
                }))
            }

            function op(a, c) {
                return ia(a).D(a, ["load"], c.O)
            }

            function Ai(a, c, b) {
                function d(k) {
                    var l;
                    return f(a, c, k) ? null === (l = h[k.dd]) || void 0 === l ? void 0 : l.nd : null
                }

                var e, f = b.Gg;
                b = b.Xi;
                var g = void 0 === b ? (e = {}, e.href = !0, e.text = !0, e) : b, h;
                return {
                    fg: function (k) {
                        return new I(function (l, m) {
                            k && k.length || m();
                            h = Gi()(k);
                            cc(a)(Sa(u({qa: [], Ba: 0}, l), function () {
                                var p = fa(a), q = p(aa), r = g.href ? rp(a, h) : [], v = g.text ? Hi(a, h) : [];
                                l({
                                    qa: Z(ca, Z(Boolean, A(d, r.concat(v)))), Ba: p(aa) -
                                        q
                                })
                            }))
                        })
                    }
                }
            }

            function rp(a, c) {
                var b = a.document.body;
                if (!b) return [];
                var d = Ii(c);
                return M(function (e, f) {
                    var g = n(f, "href");
                    try {
                        var h = decodeURI(g || "")
                    } catch (p) {
                        h = ""
                    }
                    if ("tel:" === h.slice(0, 4)) {
                        var k = (d.exec(h) || [])[0], l = k ? Ob(k) : "", m = c[l];
                        W(m) || !l && "*" !== m.nd[0] || (e.push({
                            Ge: "href",
                            Db: f,
                            dd: l,
                            ib: Ji(k, c[l].ib),
                            Gj: g
                        }), g = Ob(h.slice(4)), l = Gi()([l ? m.nd : [g, ""]]), e.push.apply(e, Hi(a, l, f)))
                    }
                    return e
                }, [], ya(b.querySelectorAll("a")))
            }

            function Hi(a, c, b) {
                void 0 === b && (b = a.document.body);
                if (!b) return [];
                var d = [], e = Ii(c);
                wf(a, b, function (f) {
                    if (f !== b && "script" !== (n(f, "parentNode.nodeName") || "").toLowerCase()) {
                        var g = Na(e.exec(f.textContent || "") || []);
                        z(function (h) {
                            var k = Ob(h);
                            W(c[k]) || d.push({Ge: "text", Db: f, dd: k, ib: Ji(h, c[k].ib), Gj: f.textContent || ""})
                        }, g)
                    }
                }, function (f) {
                    return e.test(f.textContent || "") ? 1 : 0
                }, a.NodeFilter.SHOW_TEXT);
                return d
            }

            function Gi() {
                return xd(function (a, c) {
                    var b = A(Ob, c), d = b[0];
                    b = b[1];
                    a[d] = {ib: b, nd: c};
                    var e = Ki(d);
                    e !== d && (a[e] = {ib: Ki(b), nd: c});
                    return a
                }, {})
            }

            function Ji(a, c) {
                for (var b = [], d = a.split(""),
                         e = c.split(""), f = 0, g = 0; g < a.length && !(f >= e.length); g += 1) {
                    var h = d[g];
                    "0" <= h && "9" >= h ? (b.push(e[f]), f += 1) : b.push(d[g])
                }
                return J("", b) + c.slice(f + 1)
            }

            function Ki(a) {
                var c = {7: "8", 8: "7"};
                return 11 === a.length && c[a[0]] ? "" + c[a[0]] + a.slice(1) : a
            }

            function Ii(a) {
                return new RegExp("(?:" + J("|", A(Li, da(a))) + ")")
            }

            function Mi(a, c, b, d) {
                if (c) {
                    var e = [];
                    c && (a.document.documentElement.contains(c) ? wf(a, c, qa("push", e), d) : Oa(e, Ni(a, c, d)));
                    z(b, e)
                }
            }

            function wf(a, c, b, d, e) {
                function f(g) {
                    return T(d) ? d(g) ? a.NodeFilter.FILTER_ACCEPT :
                        a.NodeFilter.FILTER_REJECT : a.NodeFilter.FILTER_ACCEPT
                }

                void 0 === e && (e = -1);
                if (T(b) && f(c) === a.NodeFilter.FILTER_ACCEPT && (b(c), !If(c))) for (c = a.document.createTreeWalker(c, e, d ? {acceptNode: f} : null, !1); c.nextNode() && !1 !== b(c.currentNode);) ;
            }

            function Ni(a, c, b) {
                var d = [], e = t(O, qa("push", d));
                T(b) ? (b = b(c), (ma(b) || b === a.NodeFilter.FILTER_ACCEPT) && e(c)) : e(c);
                if (c.childNodes && 0 < c.childNodes.length) {
                    c = c.childNodes;
                    b = 0;
                    for (var f = c.length; b < f; b += 1) {
                        var g = Ni(a, c[b]);
                        z(e, g)
                    }
                }
                return d
            }

            function Oi(a, c, b) {
                var d;
                a = [Pi(a,
                    c, function (e) {
                        d = e;
                        e.Aa.D(b)
                    }), function () {
                    d && d.unsubscribe()
                }];
                return E([Qi, a], z)
            }

            function sp(a, c, b, d) {
                var e, f;
                if (b) {
                    var g = n(d, "ecommerce") || {};
                    var h = n(d, "event") || "";
                    g = La(g) && wa(h) ? Sf(h, g) : void 0;
                    if (!g) a:{
                        var k = d;
                        !ca(d) && qe(a, Pa(d)) && (k = Ca(k));
                        if (ca(k) && (g = k[0], h = k[1], k = k[2], wa(h) && La(k) && "event" === g)) {
                            g = Sf(h, k);
                            break a
                        }
                        g = void 0
                    }
                    if (d = g || tp(d)) ib(a, {
                        da: c,
                        name: "ecommerce",
                        data: d
                    }), b((e = {}, e.__ym = (f = {}, f.ecommerce = [d], f), e))
                }
            }

            function tp(a) {
                var c = n(a, "ecommerce");
                if (La(c)) return a = Z(uc(Ri), da(c)), a =
                    M(function (b, d) {
                        b[d] = c[d];
                        return b
                    }, {}, a), 0 === da(a).length ? void 0 : a
            }

            function Sf(a, c) {
                var b, d, e = wa(a) ? yd[a] : a;
                if (e) {
                    var f = e.event, g = e.Ma, h = e.si, k = void 0 === h ? "items" : h, l = c.purchase || c;
                    if (h = l[k]) {
                        e = A(u(e.za, up), h);
                        var m = (b = {}, b[f] = g ? (d = {}, d[g] = e, d) : e, b);
                        b = da(l);
                        g && 1 < b.length && (m[f].actionField = M(function (p, q) {
                            if (q === k) return p;
                            if ("currency" === q) return m.currencyCode = l.currency, p;
                            p[vp[q] || Yf[q] || q] = l[q];
                            return p
                        }, {}, b));
                        return m
                    }
                }
            }

            function up(a, c) {
                var b = {};
                z(function (d) {
                    var e = a[d] || Yf[d] || d;
                    -1 !== d.indexOf("item_category") ?
                        (e = Yf.item_category, b[e] = b[e] ? b[e] + ("/" + c[d]) : c[d]) : b[e] = c[d]
                }, da(c));
                return b
            }

            function wp(a, c, b) {
                if (b && (b = Vf(a, b), b = Xf(a, b))) {
                    b = "?" + zd(b);
                    var d = Gb(a, c, "Button goal. Counter " + c.id + ". Button: " + b + ".");
                    te(a, c, "btn", d).reachGoal(b)
                }
            }

            function xp(a, c, b, d) {
                d = n(d, "target");
                (d = dc("button,input", a, d)) && "submit" === d.type && (d = ui(a, d)) && (b.push(d), V(a, E([!1, a, c, b, d], Si), 300))
            }

            function Si(a, c, b, d, e) {
                var f = Pb(c)(e, d), g = -1 !== f;
                if (a || g) g && d.splice(f, 1), a = vi(c, e), a = "?" + zd(a), d = E([c, b, "Form goal. Counter " + b.id + ". Form: " +
                a + "."], Ti), te(c, b, "form", d).reachGoal(a)
            }

            function Ti(a, c, b) {
                return yp(a, c).then(function (d) {
                    d && Gb(a, c, b)()
                })
            }

            function vi(a, c, b) {
                return Ui(a, c, ["i", "n", "p"], void 0, b)
            }

            function zp(a, c) {
                var b;
                a((b = {}, b.clickmap = W(c) ? !0 : c, b))
            }

            function Ap(a, c, b, d, e) {
                var f;
                c = {H: Da(), G: (f = {}, f["page-url"] = c, f["pointer-click"] = b, f), ja: {ta: "clmap/" + e.id}};
                d(c, e)["catch"](D(a, "c.s.c"))
            }

            function Bp(a, c, b, d, e) {
                if (Ad(a, "ymDisabledClickmap") || !c || !c.element) return !1;
                a = Ia(c.element);
                if (e && !e(c.element, a) || K(c.button, [2, 3]) && "A" !==
                    a || Ja(ka(a), d)) return !1;
                d = c.element;
                if (c && b) {
                    if (50 > c.time - b.time) return !1;
                    e = Math.abs(b.position.x - c.position.x);
                    a = Math.abs(b.position.y - c.position.y);
                    c = c.time - b.time;
                    if (b.element === d && 2 > e && 2 > a && 1E3 > c) return !1
                }
                for (; d;) {
                    if (Cp(d)) return !1;
                    d = d.parentElement
                }
                return !0
            }

            function Dp(a, c) {
                var b = null;
                try {
                    if (b = c.target || c.srcElement) !b.ownerDocument && b.documentElement ? b = b.documentElement : b.ownerDocument !== a.document && (b = null)
                } catch (d) {
                }
                return b
            }

            function Ep(a) {
                var c = a.which;
                a = a.button;
                return c || void 0 === a ? c : 1 ===
                a || 3 === a ? 1 : 2 === a ? 3 : 4 === a ? 2 : 0
            }

            function Vi(a, c) {
                var b = Yb(a), d = Zf(a);
                return {
                    x: c.pageX || c.clientX + d.x - (b.clientLeft || 0) || 0,
                    y: c.pageY || c.clientY + d.y - (b.clientTop || 0) || 0
                }
            }

            function ue(a, c) {
                return {
                    Z: function (b, d) {
                        var e, f = b.H, g = b.La, h = b.G, k = b.Y;
                        k = void 0 === k ? {} : k;
                        if (f && h) {
                            var l = fa(a);
                            f.fc("rqnl", 1);
                            for (var m = Bd(a), p = 1; m[p];) p += 1;
                            b.V || (b.V = {});
                            b.V.cc = p;
                            m[p] = (e = {}, e.protocol = sa.Za, e.host = bc, e.resource = b.ja.ta, e.postParams = k.fa, e.time = l(aa), e.counterType = c.ba, e.params = h, e.browserInfo = f.l(), e.counterId = c.id, e.ghid =
                                Lb(a), e);
                            g && (m[p].telemetry = g.l());
                            $f(a)
                        }
                        d()
                    }, Ea: function (b, d) {
                        Wi(a, b);
                        d()
                    }
                }
            }

            function Wi(a, c) {
                var b = Bd(a);
                c.H && !Ua(b) && c.V && (delete b[c.V.cc], $f(a))
            }

            function $f(a) {
                var c = Bd(a);
                Ra(a).C("retryReqs", c)
            }

            function Fp(a, c) {
                if (a.Oj()) {
                    var b = Xi(c);
                    if (b && !$b("ym-disable-tracklink", b)) {
                        var d = a.l, e = a.zh, f = a.kb, g = a.sender, h = a.Nh, k = f.Hc, l = b.href;
                        var m = ob(b.innerHTML && b.innerHTML.replace(/<\/?[^>]+>/gi, ""));
                        m || (m = (m = b.querySelector("img")) ? ob(m.getAttribute("title") || m.getAttribute("alt")) : "");
                        m = l === m ? "" : m;
                        if ($b("ym-external-link",
                            b)) ve(d, f, {url: l, ub: !0, title: m, sender: g}); else {
                            k = k ? Hc(d, k).hostname : S(d).hostname;
                            h = RegExp("\\.(" + J("|", A(Gp, h)) + ")$", "i");
                            var p = b.protocol + "//" + b.hostname + b.pathname;
                            h = Yi.test(p) || Yi.test(l) || h.test(l) || h.test(p);
                            b = b.hostname;
                            we(k) === we(b) ? h ? ve(d, f, {
                                url: l,
                                Pc: !0,
                                title: m,
                                sender: g
                            }) : m && e.C("il", ob(m).slice(0, 100)) : l && Hp.test(l) || ve(d, f, {
                                url: l,
                                Uc: !0,
                                ub: !0,
                                Pc: h,
                                title: m,
                                sender: g
                            })
                        }
                    }
                }
            }

            function ve(a, c, b) {
                var d, e = Da();
                b.Pc && e.C("dl", 1);
                b.ub && e.C("ln", 1);
                var f = b.Ng || {};
                e = {
                    H: e, V: {
                        title: f.title || b.title, Uc: !!b.Uc,
                        ea: f.params
                    }, G: (d = {}, d["page-url"] = b.url, d["page-ref"] = c.Hc || S(a).href, d)
                };
                d = "Link";
                b.Pc ? d = b.ub ? "Ext link - File" : "File" : b.ub && (d = "Ext link");
                ib(a, {
                    da: N(c),
                    name: "event",
                    data: {Eb: "Link click", name: (b.ub ? "external" : "internal") + " url: " + b.url}
                });
                c = b.sender(e, c).then(Gb(a, c, d + ". Counter " + c.id + ". Url: " + b.url, b.Ng));
                return Lc(a, "cl.p.s", c, f.callback || B, f.ctx)
            }

            function Ip(a, c) {
                var b, d, e = (b = {}, b.string = !0, b.object = !0, b["boolean"] = c, b)[typeof c] || !1;
                a((d = {}, d.trackLinks = e, d))
            }

            function Jp(a, c, b, d) {
                var e =
                    S(a), f = e.hostname;
                e = e.href;
                if (c = Cd(c).url) a = Hc(a, c), f = a.hostname, e = a.href;
                return [d + "://" + f + "/" + b, e || ""]
            }

            function Zi(a) {
                return (a.split(":")[1] || "").replace(/^\/*/, "").replace(/^www\./, "").split("/")[0]
            }

            function Kp(a, c, b, d) {
                var e;
                if (a = Ha(a, b)) {
                    var f = d.data;
                    b = "" + b.id;
                    var g = d.sended || [];
                    d.sended || (d.sended = g);
                    K(b, g) || !a.params || d.counter && "" + d.counter !== b || (a.params(f), g.push(b), d.parent && c.sg((e = {}, e.type = "params", e.data = f, e)))
                }
            }

            function wh(a, c, b) {
                void 0 === b && (b = O);
                var d = id(a);
                b(d);
                var e = u(d, Lp);
                xe(a,
                    c, function (f) {
                        f.Aa.D(e)
                    });
                return d
            }

            function Lp(a, c) {
                var b = n(c, "ymetrikaEvent");
                b && a.O(n(b, "type"), b)
            }

            function xe(a, c, b, d) {
                void 0 === b && (b = B);
                void 0 === d && (d = !1);
                var e = wd(a);
                if (c && T(c.push)) {
                    var f = c.push;
                    c.push = function () {
                        var g = Ca(arguments), h = g[0];
                        d && e.O(h);
                        g = f.apply(c, g);
                        d || e.O(h);
                        return g
                    };
                    a = {
                        Aa: e, unsubscribe: function () {
                            c.push = f
                        }
                    };
                    b(a);
                    z(e.O, c);
                    return a
                }
            }

            function Zd(a) {
                a = H(a);
                var c = a.o("dataLayer", []);
                a.C("dataLayer", c);
                return c
            }

            function bm(a, c) {
                var b, d;
                a.push((b = {}, b.ymetrikaEvent = (d = {}, d.type = c,
                    d), b))
            }

            function $i(a, c) {
                var b = cd(a, c), d = [], e = [];
                if (!b) return null;
                var f = E([a, b.Je], Mp), g = u(f, Np);
                b.ca.D(["initToParent"], function (h) {
                    g(d, b.children[h[1].counterId])
                }).D(["parentConnect"], function (h) {
                    g(e, b.Ja[h[1].counterId])
                });
                return {
                    ca: b.ca, Nk: function (h, k) {
                        return new I(function (l, m) {
                            b.Je(h, k, function (p, q) {
                                l([p, q])
                            });
                            V(a, u(Ta(), m), 5100, "is.o")
                        })
                    }, rg: function (h) {
                        var k = {vg: [], Ue: [], data: h};
                        d.push(k);
                        return f(b.children, k, h)
                    }, sg: function (h) {
                        var k = {vg: [], Ue: [], data: h};
                        e.push(k);
                        return f(b.Ja, k, h)
                    }
                }
            }

            function Np(a, c, b) {
                c = Z(function (d) {
                    return !K(b.info.counterId, d.Ue)
                }, c);
                z(function (d) {
                    var e;
                    b.info.counterId && a((e = {}, e[b.info.counterId] = b, e), d, d.data)
                }, c)
            }

            function Mp(a, c, b, d, e) {
                return (new I(function (f, g) {
                    var h = da(b), k = t(d.resolve ? d.resolve : O, gd(f)), l = t(d.reject ? d.reject : O, gd(g));
                    d.resolve = k;
                    d.reject = l;
                    z(function (m) {
                        var p;
                        d.Ue.push(+m);
                        var q = b[m], r = V(a, u(Ta(), l), 5100, "is.m");
                        c(q.window, y(e, (p = {}, p.toCounter = Ga(m), p)), function (v, w) {
                            na(a, r);
                            d.vg.push(m);
                            d.resolve && d.resolve(w)
                        })
                    }, h)
                }))["catch"](D(a,
                    "if.b"))
            }

            function Op(a) {
                var c = B, b = null, d = a.length;
                if (0 !== a.length && a[0]) {
                    var e = a.slice(-1)[0];
                    T(e) && (c = e, d = a.length + -1);
                    var f = a.slice(-2)[0];
                    T(f) && (c = f, b = e, d = a.length + -2);
                    d = a.slice(0, d);
                    return {Ah: b, pc: c, ea: 1 === d.length ? a[0] : Gc(d)}
                }
            }

            function Lc(a, c, b, d, e) {
                var f = E([a, d, e], ag);
                return b.then(f, function (g) {
                    f();
                    oe(a, c, g)
                })
            }

            function aj(a, c) {
                return {
                    Z: function (b, d) {
                        var e = (b.V || {}).ea, f = b.Y;
                        f = void 0 === f ? {} : f;
                        if (e && (bj(c, e), !f.fa && b.H && b.G)) {
                            var g = mb(a, e), h = cj(a), k = b.H.o("pv");
                            g && !b.G.nohit && (ib(a, {
                                da: N(c),
                                name: "params", data: {Sk: e}
                            }), k ? encodeURIComponent(g).length > sa.Wg ? h.push([b.H, e]) : b.G["site-info"] = g : (f.fa = g, b.Y = f, b.Ta || (b.Ta = {}), b.Ta.Yi = !0))
                        }
                        d()
                    }, Ea: function (b, d) {
                        var e = cj(a), f = Ha(a, c), g = f && f.params;
                        g && (f = Z(t(xc, ka(b.H)), e), z(function (h) {
                            g(h[1]);
                            h = ye(a)(h, e);
                            e.splice(h, 1)
                        }, f));
                        d()
                    }
                }
            }

            function ze(a, c) {
                return function (b) {
                    bg(a, c, b)
                }
            }

            function Pp(a, c) {
                cg(a)(function (b) {
                    delete b[c]
                })
            }

            function bg(a, c, b) {
                cg(a)(function (d) {
                    d[c] = y(d[c] || {}, b)
                })
            }

            function cg(a) {
                a = H(a);
                var c = a.o("dsjf") || za({});
                a.Va("dsjf", c);
                return c
            }

            function Qp(a, c) {
                return function (b) {
                    var d, e, f = Ha(a, c);
                    f && (La(b) ? Pa(da(b)) ? (b = dj(b)) && Pa(b) && f.params((d = {}, d.__ym = (e = {}, e.fpmh = b, e), d)) : Gb(a, c, "First party params error. Empty object.")() : Gb(a, c, "First party params error. Not an object.")())
                }
            }

            function dj(a) {
                return M(function (c, b) {
                    var d = b[0], e = b[1], f = La(e);
                    if (!wa(e) && !f) return c;
                    e = f ? dj(e) : e;
                    Pa(e) && c.push([d, e]);
                    return c
                }, [], pa(a))
            }

            function ej(a, c, b) {
                void 0 === b && (b = 0);
                c = pa(c);
                c = M(function (d, e) {
                    var f = e[0], g = e[1], h = La(g);
                    if (!wa(g) && !h) return d;
                    h ? g = ej(a, g, b + 1) : b || "yandex_cid" !== f ? ("phone_number" === f ? g = Rp(g) : "email" === f && (g = Sp(g)), g = fj(a, g)) : g = I.resolve(g);
                    d.push(g.then(function (k) {
                        return [f, k]
                    }));
                    return d
                }, [], c);
                return I.all(c)
            }

            function Sp(a) {
                var c = ob(a).toLowerCase().split("@"), b = c[0];
                c = c[1];
                if (!c) return a;
                c = c.replace("googlemail.com", "gmail.com");
                gj(c) && (c = "yandex.ru");
                "yandex.ru" === c ? b = b.replace(dg, "-") : "gmail.com" === c && (b = b.replace(dg, ""));
                a = he(b, "+");
                -1 !== a && (b = b.slice(0, a));
                return b + "@" + c
            }

            function Rp(a) {
                a = Ob(a);
                return "8" === a[0] ? "7" +
                    a.slice(1) : a
            }

            function fj(a, c) {
                return new I(function (b, d) {
                    var e = (new a.TextEncoder).encode(c);
                    a.crypto.subtle.digest("SHA-256", e).then(function (f) {
                        f = new a.Blob([f], {type: "application/octet-binary"});
                        var g = new a.FileReader;
                        g.onload = function (h) {
                            h = n(h, "target.result");
                            var k = (h || "").indexOf(",");
                            -1 !== k ? b(h.substring(k + 1)) : d(ic("fpm.i"))
                        };
                        g.readAsDataURL(f)
                    }, d)
                })
            }

            function Ha(a, c) {
                var b = H(a).o("counters", {}), d = N(c);
                return b[d]
            }

            function hj(a, c) {
                H(a).C("dce:" + c, !0);
                var b = H(a).o("dclq:" + c);
                b && (z(function (d) {
                    var e =
                        d[0];
                    d = d[1];
                    ha.apply(void 0, xa([Dd(a, c)[e]], d))
                }, b), Ed(b))
            }

            function Gb(a, c, b, d) {
                return eg(c) ? B : u(E(xa([a, N(c)], d ? [b + ". Params:", d] : [b]), Db), ha)
            }

            function Db() {
                var a = Ca(arguments), c = a.slice(2);
                Dd(a[0], a[1]).log.apply(Db, c)
            }

            function Jf(a) {
                var c = "1" === ac(a).o("debug"), b = Ae(a, "1") || Ae(a, "2");
                a = a._ym_debug;
                return {yi: c, Ii: a || b, isEnabled: Ja(Boolean, [c, a, b])}
            }

            function Ae(a, c) {
                return -1 < S(a).href.indexOf("_ym_debug=" + c)
            }

            function Tp(a, c) {
                return {log: Mc(a, "log", c, B), warn: Mc(a, "log", c, B), error: Mc(a, "log", c, B)}
            }

            function Mc(a, c, b, d) {
                return function () {
                    var e = Ca(arguments);
                    ib(a, {da: b, name: "log", data: {Nb: e, type: c}});
                    return d.apply(void 0, e)
                }
            }

            function ra(a, c) {
                var b = N(a);
                return ij()(Up(b)).then(c)
            }

            function Vp(a, c, b) {
                c = N(c);
                var d = fg(a);
                b = y({Ph: d(aa)}, b);
                ib(a, {da: c, name: "counterSettings", data: {Pk: b}});
                return ij()(Wp(c, b))
            }

            function Wp(a, c) {
                return function (b) {
                    var d = b[a];
                    d ? (d.rj = c, d.ig = !0, d.hg ? d.hg(c) : d.gb = I.resolve(c)) : b[a] = {
                        gb: I.resolve(c),
                        rj: c,
                        ig: !0
                    }
                }
            }

            function gg(a) {
                return !pd(a) && hg(a)
            }

            function Fd(a) {
                return ab(a) ?
                    u(a, Xp) : !1
            }

            function hg(a) {
                var c = n(a, "navigator.sendBeacon");
                return c && Aa("sendBeacon", c) ? E([a, F(c, n(a, "navigator"))], Yp) : !1
            }

            function Yp(a, c, b, d) {
                return new I(function (e, f) {
                    var g;
                    if (!n(a, "navigator.onLine")) return f();
                    var h = y(d.bc, (g = {}, g["force-urlencoded"] = 1, g));
                    g = b + "?" + zd(h) + (d.fa ? "&" + d.fa : "");
                    return 2E3 < g.length ? f(Ta("sb.tlq")) : c(g) ? e("") : f()
                })
            }

            function Xp(a, c, b) {
                return new I(function (d, e) {
                    var f, g, h = "_ymjsp" + Va(a), k = y((f = {}, f.callback = h, f), b.bc), l = E([a, h], Zp);
                    a[h] = function (p) {
                        try {
                            l(), nc(m), d(p)
                        } catch (q) {
                            e(q)
                        }
                    };
                    k.wmode = "5";
                    var m = lc(a, (g = {}, g.src = jj(c, b, k), g));
                    if (!m) return l(), e(ic("jp.s"));
                    f = u(m, nc);
                    f = t(f, u(Ta(b.Fa), e));
                    g = Be(a, f, b.Ib || 1E4);
                    g = E([a, g], na);
                    m.onload = g;
                    m.onerror = t(l, g, f)
                })
            }

            function Zp(a, c) {
                try {
                    delete a[c]
                } catch (b) {
                    a[c] = void 0
                }
            }

            function Nc(a) {
                var c = ab(a);
                return c ? E([a, c], $p) : !1
            }

            function $p(a, c, b, d) {
                return new I(function (e, f) {
                    var g = Yb(a), h = c("img"), k = t(u(h, nc), u(Ta(d.Fa), f)), l = Be(a, k, d.Ib || 3E3);
                    h.onerror = k;
                    h.onload = t(u(h, nc), u(null, e), E([a, l], na));
                    k = y({}, d.bc);
                    delete k.wmode;
                    h.src = jj(b, d, k);
                    rd(a) &&
                    (y(h.style, {
                        position: "absolute",
                        visibility: "hidden",
                        width: "0px",
                        height: "0px"
                    }), g.appendChild(h))
                })
            }

            function wb(a) {
                var c;
                if (c = n(a, "XMLHttpRequest")) if (c = "withCredentials" in new a.XMLHttpRequest) {
                    a:{
                        if (aq.test(a.location.host) && a.opera && T(a.opera.version) && (c = a.opera.version(), "string" === typeof c && "12" === c.split(".")[0])) {
                            c = !0;
                            break a
                        }
                        c = !1
                    }
                    c = !c
                }
                return c ? u(a, bq) : !1
            }

            function bq(a, c, b) {
                var d, e = new a.XMLHttpRequest, f = b.fa, g = y(b.qd ? (d = {}, d.wmode = "7", d) : {}, b.bc);
                return new I(function (h, k) {
                    e.open(b.Ze || "GET",
                        c + "?" + zd(g), !0);
                    e.withCredentials = !1 !== b.Pg;
                    b.Ib && (e.timeout = b.Ib);
                    kj(pa, hb(function (m) {
                        e.setRequestHeader(m[0], m[1])
                    }))(b.Cb);
                    var l = E([a, e, Ta(b.Fa), b.qd, b.uj, h, k], cq);
                    e.onreadystatechange = l;
                    try {
                        e.send(f)
                    } catch (m) {
                    }
                })
            }

            function cq(a, c, b, d, e, f, g, h) {
                if (4 === c.readyState) if (200 === c.status || e || g(b), e) 200 === c.status ? f(c.responseText) : g(ic("http." + c.status + ".st." + c.statusText + ".rt." + ("" + c.responseText).substring(0, 50))); else {
                    e = null;
                    if (d) try {
                        (e = tb(a, c.responseText)) || g(b)
                    } catch (k) {
                        g(b)
                    }
                    f(e)
                }
                return h
            }

            function jj(a,
                        c, b) {
                (b = zd(b)) && (a += "?" + b);
                c.fa && (a += (b ? "&" : "?") + c.fa);
                return a
            }

            function dq(a, c, b) {
                var d = A(xc, Qb[c] || Rb);
                z(function (e) {
                    return d.unshift(e)
                }, Ce);
                return A(t(Oc([a, b]), ha), d)
            }

            function lj(a, c) {
                var b = S(a), d = b.href, e = b.host, f = -1;
                if (!wa(c) || W(c)) return d;
                b = c.replace(mj, "");
                if (-1 !== b.search(eq)) return b;
                var g = b.charAt(0);
                if ("?" === g && (f = d.search(/\?/), -1 === f) || "#" === g && (f = d.search(/#/), -1 === f)) return d + b;
                if (-1 !== f) return d.substr(0, f) + b;
                if ("/" === g) {
                    if (f = d.indexOf(e), -1 !== f) return d.substr(0, f + e.length) + b
                } else return d =
                    d.split("/"), d[d.length - 1] = b, J("/", d);
                return ""
            }

            function De(a, c) {
                return {
                    Z: function (b, d) {
                        var e = nj(c);
                        e = E([b, e, d], fq);
                        gq(a, c, e)
                    }, Ea: function (b, d) {
                        var e = b.H, f = nj(c);
                        if (e) {
                            var g = f.wa;
                            f.wf === e && g && (z(ha, g), f.wa = null)
                        }
                        d()
                    }
                }
            }

            function fq(a, c, b) {
                var d = a.H;
                d ? re(a) ? (c.wf = d, b()) : c.wa ? c.wa.push(b) : b() : b()
            }

            function re(a) {
                return (a = a.H) && a.o("pv") && !a.o("ar")
            }

            function gq(a, c, b) {
                if (ig(a) && eb(a)) {
                    var d = hq(c);
                    if (!d.Ki) {
                        d.Ki = !0;
                        c = cd(a, c);
                        if (!c) {
                            b();
                            return
                        }
                        d.wa = [];
                        var e = function () {
                            d.wa && (z(ha, d.wa), d.wa = null)
                        };
                        V(a, e, 3E3);
                        c.ca.D(["initToChild"], e)
                    }
                    d.wa ? d.wa.push(b) : b()
                } else b()
            }

            function oj(a, c) {
                return {
                    Z: function (b, d) {
                        var e = b.H;
                        if (e && (!c || c.qg)) {
                            var f = a.document.title;
                            b.V && b.V.title && (f = b.V.title);
                            var g = ec("getElementsByTagName", a.document);
                            "string" !== typeof f && g && (f = g("title"), f = (f = n(f, "0.innerHtml")) ? f : "");
                            f = f.slice(0, sa.Xg);
                            e.C("t", f)
                        }
                        d()
                    }
                }
            }

            function Hb(a) {
                return function (c, b) {
                    return {
                        Z: function (d, e) {
                            var f = d.H, g = d.G;
                            f && g && z(function (h) {
                                var k = Ee[h], l = "bi", m = f;
                                k || (k = Gd[h], l = "tel", m = $d(d));
                                k && (k = C(l + ":" + h, k, null)(c, b,
                                    d), m.fc(h, k))
                            }, a || iq());
                            e()
                        }
                    }
                }
            }

            function jq(a, c) {
                var b = Hd(a);
                c.D(["initToParent"], function (d) {
                    var e = d[0];
                    d = d[1];
                    window.window && (b.children[d.counterId] = {info: d, window: e.source})
                }).D(["initToChild"], function (d) {
                    var e = d[0];
                    d = d[1];
                    e.source === a.parent && c.O("parentConnect", [e, d])
                }).D(["parentConnect"], function (d) {
                    var e = d[1];
                    e.counterId && (b.Ja[e.counterId] = {info: e, window: d[0].source})
                })
            }

            function kq(a) {
                if (Aa("MutationObserver", a.MutationObserver)) {
                    var c = Hd(a).children, b = new a.MutationObserver(function () {
                        z(function (d) {
                            n(c[d],
                                "window.window") || delete c[d]
                        }, da(c))
                    });
                    cc(a)(Sa(B, function () {
                        b.observe(a.document.body, {subtree: !0, childList: !0})
                    }))
                }
            }

            function lq(a, c) {
                return function (b, d) {
                    var e, f = {Bc: fa(a)(aa), key: a.Math.random(), dir: 0};
                    b.length && (f.Bc = Ga(b[0]), f.key = parseFloat(b[1]), f.dir = Ga(b[2]));
                    y(d, c);
                    var g = (e = {data: d}, e.__yminfo = J(":", ["__yminfo", f.Bc, f.key, f.dir]), e);
                    return {aa: f, Ag: mb(a, g) || ""}
                }
            }

            function cc(a, c) {
                function b(e) {
                    n(c, d) ? e() : V(a, u(e, b), 100)
                }

                void 0 === c && (c = a);
                var d = (c.nodeType ? "contentWindow." : "") + "document.body";
                return za(function (e, f) {
                    b(f)
                })
            }

            function Fc(a, c) {
                var b = c.je, d = b || "uid";
                b = b ? a.location.hostname : void 0;
                var e = ac(a), f = Ra(a), g = fa(a)(jg), h = pj(a, c), k = h[0];
                h = h[1];
                var l = e.o("d");
                qj(a, c);
                var m = !1;
                !h && k && (h = k, m = !0);
                if (!h) h = J("", [g, Va(a, 1E6, 999999999)]), m = !0; else if (!l || 15768E3 < g - Ga(l)) m = !0;
                m && !c.xb && (e.C(d, h, 525600, b), e.C("d", "" + g, 525600, b));
                f.C(d, h);
                return h
            }

            function mq(a, c) {
                return !c.xb && qj(a, c)
            }

            function pj(a, c) {
                var b = Ra(a), d = ac(a), e = c.je || "uid";
                return [b.o(e), d.o(e)]
            }

            function ai(a, c, b) {
                kg(a, "metrika_enabled",
                    "1", 0, c, b, !0);
                var d = rj(a);
                (d = d && d.metrika_enabled) && sj(a, "metrika_enabled", c, b, !0);
                return !!d
            }

            function kg(a, c, b, d, e, f, g) {
                void 0 === g && (g = !1);
                if (bi(a, je, c)) {
                    var h = c + "=" + encodeURIComponent(b) + ";";
                    h += "" + nq(a);
                    if (d) {
                        var k = new Date;
                        k.setTime(k.getTime() + 6E4 * d);
                        h += "expires=" + k.toUTCString() + ";"
                    }
                    e && (d = e.replace(oq, ""), h += "domain=" + d + ";");
                    try {
                        a.document.cookie = h + ("path=" + (f || "/")), g || (tj(a)[c] = b)
                    } catch (l) {
                    }
                }
            }

            function je(a, c) {
                var b = tj(a);
                return b ? b[c] || null : null
            }

            function rj(a) {
                try {
                    var c = a.document.cookie;
                    if (!ma(c)) {
                        var b =
                            {};
                        z(function (d) {
                            d = d.split("=");
                            var e = d[1];
                            b[ob(d[0])] = ob(uj(e))
                        }, (c || "").split(";"));
                        return b
                    }
                } catch (d) {
                }
                return null
            }

            function bi(a, c, b) {
                return !lg.length || K(b, mg) ? !0 : M(function (d, e) {
                    return d && e(a, c, b)
                }, !0, lg)
            }

            function Lb(a) {
                var c = H(a), b = c.o("hitId");
                b || (b = Va(a), c.C("hitId", b));
                return b
            }

            function gj(a) {
                var c = a.match(vj);
                if (c) {
                    a = c[1];
                    if (c = c[2]) return K(c, ng) ? c : !1;
                    if (a) return ng[0]
                }
                return !1
            }

            function S(a) {
                return M(function (c, b) {
                    var d = n(a, "location." + b);
                    c[b] = d ? "" + d : "";
                    return c
                }, {}, pq)
            }

            function Xf(a, c, b) {
                var d =
                    Ia(c);
                return d && Ui(a, c, Na(["p", qq[d], "c"]), Wf, b)
            }

            function Vf(a, c) {
                var b = dc(og, a, c);
                if (!b) {
                    var d = dc("div", a, c);
                    d && (xb(og + ",div", d).length || (b = d))
                }
                return b
            }

            function Ui(a, c, b, d, e) {
                return M(function (f, g) {
                    var h = null;
                    g in wj ? h = c.getAttribute && c.getAttribute(wj[g]) : g in Pc && (h = "p" === g ? Pc[g](a, c, e) : "c" === g ? Pc[g](a, c, d) : Pc[g](a, c));
                    h && (h = h.slice(0, xj[g] || 100), f[g] = pg[g] ? "" + oc(h) : h);
                    return f
                }, {}, b)
            }

            function Th(a, c, b) {
                if (Id(a)) return ya(b.querySelectorAll(c));
                var d = yj(c.split(" "), b);
                return Z(function (e, f) {
                    return Pb(a)(e,
                        d) === f
                }, d)
            }

            function yj(a, c) {
                var b = xa(a), d = b.shift();
                if (!d) return [];
                d = c.getElementsByTagName(d);
                return b.length ? mc(u(b, yj), ya(d)) : ya(d)
            }

            function qc(a, c) {
                if (c.querySelector) return c.querySelector(a);
                var b = xb(a, c);
                return b && b.length ? b[0] : null
            }

            function xb(a, c) {
                if (!c || !c.querySelectorAll) return [];
                var b = c.querySelectorAll(a);
                return b ? ya(b) : []
            }

            function Xi(a) {
                var c = null;
                try {
                    c = a.target || a.srcElement
                } catch (b) {
                }
                if (c) {
                    3 === c.nodeType && (c = c.parentNode);
                    for (a = c && c.nodeName && ("" + c.nodeName).toLowerCase(); n(c, "parentNode.nodeName") &&
                    ("a" !== a && "area" !== a || !c.href && !c.getAttribute("xlink:href"));) a = (c = c.parentNode) && c.nodeName && ("" + c.nodeName).toLowerCase();
                    return c.href ? c : null
                }
                return null
            }

            function lc(a, c) {
                var b = a.document, d = y({type: "text/javascript", charset: "utf-8", async: !0}, c), e = ab(a);
                if (e) {
                    var f = e("script");
                    kj(pa, hb(function (l) {
                        var m = l[0];
                        l = l[1];
                        "async" === m && l ? f.async = !0 : f[m] = l
                    }))(d);
                    try {
                        var g = ec("getElementsByTagName", b), h = g("head")[0];
                        if (!h) {
                            var k = g("html")[0];
                            h = e("head");
                            k && k.appendChild(h)
                        }
                        h.insertBefore(f, h.firstChild);
                        return f
                    } catch (l) {
                    }
                }
            }

            function rq(a, c, b) {
                var d = zj(c);
                K(b, d.zb) || d.zb.push(b);
                if (Ua(d.qb)) {
                    b = ab(a);
                    if (!b) return null;
                    b = b("iframe");
                    y(b.style, {display: "none", width: "1px", height: "1px", visibility: "hidden"});
                    b.src = c;
                    a = Yb(a);
                    if (!a) return null;
                    a.appendChild(b);
                    d.qb = b
                } else (a = n(d.qb, "contentWindow")) && a.postMessage("frameReinit", "*");
                return d.qb
            }

            function sq(a, c) {
                var b = ca(a) ? a : [a];
                c = c || document;
                if (c.querySelectorAll) {
                    var d = J(", ", A(function (e) {
                        return "." + e
                    }, b));
                    return ya(c.querySelectorAll(d))
                }
                if (c.getElementsByClassName) return mc(t(qa("getElementsByClassName",
                    c), ya), b);
                d = c.getElementsByTagName("*");
                b = "(" + J("|", b) + ")";
                return Z(u(b, $b), ya(d))
            }

            function qg(a, c, b) {
                for (var d = "", e = Uf(), f = Ia(c) || "*"; c && c.parentNode && !K(f, ["BODY", "HTML"]);) d += e[f] || "*", d += Aj(a, c, b) || "", c = c.parentElement, f = Ia(c) || "*";
                return ob(d, 128)
            }

            function Aj(a, c, b) {
                if (a = Fe(a, c)) {
                    a = a.childNodes;
                    for (var d = c && c.nodeName, e = 0, f = 0; f < a.length; f += 1) if (d === (a[f] && a[f].nodeName)) {
                        if (c === a[f]) return e;
                        b && a[f] === b || (e += 1)
                    }
                }
                return 0
            }

            function Fe(a, c) {
                var b = n(a, "document");
                if (!c || c === b.documentElement) return null;
                if (c === yc(a)) return b.documentElement;
                b = null;
                try {
                    b = c.parentNode
                } catch (d) {
                }
                return b
            }

            function Ff(a, c) {
                var b = rg(a, c), d = b.left;
                b = b.top;
                var e = Ge(a, c);
                return [d, b, e[0], e[1]]
            }

            function Ge(a, c) {
                var b = n(a, "document");
                if (c === yc(a) || c === b.documentElement) {
                    b = Yb(a);
                    var d = Qc(a);
                    return [Math.max(b.scrollWidth, d[0]), Math.max(b.scrollHeight, d[1])]
                }
                return (b = Ic(c)) ? [b.width, b.height] : [c.offsetWidth, c.offsetHeight]
            }

            function rg(a, c) {
                var b = c, d = n(a, "document"), e = Ia(b);
                if (!b || !b.ownerDocument || "PARAM" === e || b === yc(a) || b === d.documentElement) return {
                    left: 0,
                    top: 0
                };
                if (d = Ic(b)) return b = Zf(a), {left: Math.round(d.left + b.x), top: Math.round(d.top + b.y)};
                for (e = d = 0; b;) d += b.offsetLeft, e += b.offsetTop, b = b.offsetParent;
                return {left: d, top: e}
            }

            function ob(a, c) {
                if (a) {
                    var b = Bj ? Bj.call(a) : ("" + a).replace(mj, "");
                    return c && b.length > c ? b.substring(0, c) : b
                }
                return ""
            }

            function dc(a, c, b) {
                if (!(c && c.Element && c.Element.prototype && c.document && b)) return null;
                if (c.Element.prototype.closest && Aa("closest", c.Element.prototype.closest) && b.closest) return b.closest(a);
                var d = $h(c);
                if (d) {
                    for (; b &&
                           1 === b.nodeType && !d.call(b, a);) b = b.parentElement || b.parentNode;
                    return b && 1 === b.nodeType ? b : null
                }
                if (Id(c)) {
                    for (a = ya((c.document || c.ownerDocument).querySelectorAll(a)); b && 1 === b.nodeType && -1 === Pb(c)(b, a);) b = b.parentElement || b.parentNode;
                    return b && 1 === b.nodeType ? b : null
                }
                return null
            }

            function Id(a) {
                return !(!Aa("querySelectorAll", n(a, "Element.prototype.querySelectorAll")) || !a.document.querySelectorAll)
            }

            function Cj(a, c, b) {
                var d = c.top, e = c.bottom, f = c.left, g = b.pd;
                b = b.Qa;
                a = a.Math;
                c = a.min(a.max(c.right, 0), g) - a.min(a.max(f,
                    0), g);
                return (a.min(a.max(e, 0), b) - a.min(a.max(d, 0), b)) * c
            }

            function Dj(a) {
                return He(a) && !Ja(ka(a.type), tq) ? Ie(a) ? !a.checked : !a.value : uq(a) ? !a.value : vq(a) ? 0 > a.selectedIndex : !0
            }

            function Ia(a) {
                if (a) try {
                    var c = a.nodeName;
                    if (wa(c)) return c;
                    c = a.tagName;
                    if (wa(c)) return c
                } catch (b) {
                }
            }

            function Ej(a, c) {
                var b = a.document.getElementsByTagName("form");
                return Pb(a)(c, ya(b))
            }

            function wq(a, c, b) {
                b = ec("dispatchEvent", b || a.document);
                var d = null, e = n(a, "Event.prototype.constructor");
                if (e && (Aa("(Event|Object|constructor)", e) ||
                    sg(a) && "[object Event]" === "" + e)) try {
                    d = new a.Event(c)
                } catch (f) {
                    if ((a = ec("createEvent", n(a, "document"))) && T(a)) {
                        try {
                            d = a(c)
                        } catch (g) {
                        }
                        d && d.initEvent && d.initEvent(c, !1, !1)
                    }
                }
                d && b(d)
            }

            function Ic(a) {
                try {
                    return a.getBoundingClientRect && a.getBoundingClientRect()
                } catch (c) {
                    return "object" === typeof c && null !== c && 16389 === (c.Yf && c.Yf & 65535) ? {
                        top: 0,
                        bottom: 0,
                        left: 0,
                        width: 0,
                        height: 0,
                        right: 0
                    } : null
                }
            }

            function Zf(a) {
                var c = yc(a), b = n(a, "document");
                return {
                    x: a.pageXOffset || b.documentElement && b.documentElement.scrollLeft || c &&
                        c.scrollLeft || 0,
                    y: a.pageYOffset || b.documentElement && b.documentElement.scrollTop || c && c.scrollTop || 0
                }
            }

            function Qc(a) {
                var c = Je(a);
                if (c) {
                    var b = c[2];
                    return [a.Math.round(c[0] * b), a.Math.round(c[1] * b)]
                }
                c = Yb(a);
                return [n(c, "clientWidth") || a.innerWidth, n(c, "clientHeight") || a.innerHeight]
            }

            function Je(a) {
                var c = n(a, "visualViewport.width"), b = n(a, "visualViewport.height");
                a = n(a, "visualViewport.scale");
                return ma(c) || ma(b) ? null : [Math.floor(c), Math.floor(b), a]
            }

            function Yb(a) {
                var c = n(a, "document") || {}, b = c.documentElement;
                return "CSS1Compat" === c.compatMode ? b : yc(a) || b
            }

            function yc(a) {
                a = n(a, "document");
                try {
                    return a.getElementsByTagName("body")[0]
                } catch (c) {
                    return null
                }
            }

            function $b(a, c) {
                try {
                    return (new RegExp("(?:^|\\s)" + a + "(?:\\s|$)")).test(c.className)
                } catch (b) {
                    return !1
                }
            }

            function zc(a) {
                var c;
                try {
                    if (c = a.target || a.srcElement) !c.ownerDocument && c.documentElement ? c = c.documentElement : c.ownerDocument !== document && (c = null)
                } catch (b) {
                }
                return c
            }

            function nc(a) {
                var c = a && a.parentNode;
                c && c.removeChild(a)
            }

            function Sb(a) {
                return a ? a.innerText ||
                    "" : ""
            }

            function If(a) {
                if (ma(a)) return !1;
                a = a.nodeType;
                return 3 === a || 8 === a
            }

            function sf(a, c, b) {
                void 0 === c && (c = "");
                void 0 === b && (b = "_ym");
                var d = "" + b + c + "_";
                return {
                    $d: xq(a), o: function (e, f) {
                        var g = Fj(a, "" + d + e);
                        return Ua(g) && !W(f) ? f : g
                    }, C: function (e, f) {
                        Gj(a, "" + d + e, f);
                        return this
                    }, Rb: function (e) {
                        Hj(a, "" + d + e);
                        return this
                    }
                }
            }

            function Gj(a, c, b) {
                var d = tg(a);
                a = mb(a, b);
                if (!Ua(a)) try {
                    d.setItem(c, a)
                } catch (e) {
                }
            }

            function Fj(a, c) {
                var b = tg(a);
                try {
                    return tb(a, b.getItem(c))
                } catch (d) {
                }
                return null
            }

            function Hj(a, c) {
                var b = tg(a);
                try {
                    b.removeItem(c)
                } catch (d) {
                }
            }

            function tg(a) {
                try {
                    return a.localStorage
                } catch (c) {
                }
                return null
            }

            function mb(a, c, b) {
                try {
                    return a.JSON.stringify(c, null, b)
                } catch (d) {
                    return null
                }
            }

            function $d(a, c, b) {
                void 0 === b && (b = null);
                a.La || (a.La = ug());
                c && a.La.fc(c, b);
                return a.La
            }

            function Ke(a) {
                return {
                    Z: function (c, b) {
                        var d = a.document, e = c.H;
                        if (e && vg(a)) {
                            var f = ia(a), g = function (h) {
                                vg(a) || (f.kc(d, Ij, g), b());
                                return h
                            };
                            f.D(d, Ij, g);
                            e.C("pr", "1")
                        } else b()
                    }
                }
            }

            function wg(a) {
                return function (c, b, d) {
                    return function (e, f) {
                        var g = Oa(Pf(c,
                            a, f), d);
                        return ne(c, b, g)(e)
                    }
                }
            }

            function ne(a, c, b) {
                var d = Eb(a, c);
                return function (e) {
                    return Jj(b, e, !0).then(function () {
                        var f = e.ja || {}, g = f.Bi, h = void 0 === g ? "" : g;
                        g = f.ta;
                        var k = void 0 === g ? "" : g;
                        f = f.Ci;
                        f = A(function (l) {
                            return sa.Za + "//" + ("" + h + l || bc) + "/" + k
                        }, void 0 === f ? [bc] : f);
                        return d(e, f)
                    }).then(function (f) {
                        var g = f.fd;
                        f = f.Lg;
                        e.tj = g;
                        e.Mk = f;
                        return Jj(b, e).then(u(g, O))
                    })
                }
            }

            function Eb(a, c) {
                return function (b, d) {
                    return Kj(a, c, d, b)
                }
            }

            function Kj(a, c, b, d, e, f) {
                var g;
                void 0 === e && (e = 0);
                void 0 === f && (f = 0);
                var h = y({Fa: []},
                    d.Y), k = c[f], l = k[0];
                k = k[1];
                var m = b[e];
                h.Cb && h.Cb["Content-Type"] || !h.fa || (h.Cb = y({}, h.Cb, (g = {}, g["Content-Type"] = "application/x-www-form-urlencoded", g)), h.fa = "site-info=" + fe(h.fa));
                h.Ze = h.fa ? "POST" : "GET";
                h.bc = yq(a, d, l);
                h.ta = (d.ja || {}).ta;
                h.Fa.push(l);
                y(d.Y, h);
                g = "" + m + (d.Ta && d.Ta.Yi ? "/1" : "");
                var p = 0;
                p = zq(a, g, h);
                return k(g, h).then(function (q) {
                    ib(a, {name: "requestSuccess", data: {body: q, requestId: p}});
                    return {fd: q, Lg: e}
                })["catch"](function (q) {
                    ib(a, {name: "requestFail", data: {error: q, requestId: p}});
                    var r = f + 1 >=
                        c.length, v = e + 1 >= b.length;
                    r && v && Xa(q);
                    return Kj(a, c, b, d, !v && r ? e + 1 : e, r ? 0 : f + 1)
                })
            }

            function yq(a, c, b) {
                var d = y({}, c.G);
                a = fa(a);
                c.H && (d["browser-info"] = Da(c.H.l()).C("st", a(jg)).Ca());
                !d.t && (c = c.La) && (c.C("ti", b), d.t = c.Ca());
                return d
            }

            function zq(a, c, b) {
                var d = Va(a);
                ib(a, {name: "request", data: {url: c, requestId: d, Ok: b}});
                return d
            }

            function zd(a) {
                return a ? t(pa, xd(function (c, b) {
                    var d = b[0], e = b[1];
                    W(e) || ma(e) || c.push(d + "=" + fe(e));
                    return c
                }, []), qd("&"))(a) : ""
            }

            function Aq(a) {
                return a ? t(hb(function (c) {
                    c = c.split("=");
                    var b = c[1];
                    return [c[0], ma(b) ? void 0 : uj(b)]
                }), xd(function (c, b) {
                    c[b[0]] = b[1];
                    return c
                }, {}))(a.split("&")) : {}
            }

            function uj(a) {
                var c = "";
                try {
                    c = decodeURIComponent(a)
                } catch (b) {
                }
                return c
            }

            function fe(a) {
                try {
                    return encodeURIComponent(a)
                } catch (c) {
                }
                a = J("", Z(function (c) {
                    return 55296 >= c.charCodeAt(0)
                }, a.split("")));
                return encodeURIComponent(a)
            }

            function Pf(a, c, b) {
                return A(t(xc, Oc([a, b]), ha), Lj[c] || [])
            }

            function Mj(a, c, b, d) {
                a[c] || (a[c] = []);
                b && !ma(d) && Nj(a[c], b, d)
            }

            function Nj(a, c, b) {
                for (var d = [c, b], e = -1E4, f = 0; f < a.length; f +=
                    1) {
                    var g = a[f], h = g[0];
                    g = g[1];
                    if (b === g && h === c) return;
                    if (b < g && b >= e) {
                        a.splice(f, 0, d);
                        return
                    }
                    e = g
                }
                a.push(d)
            }

            function fc(a) {
                z(function (c) {
                    var b = c[1];
                    gf[c[0]] = {ia: b.ia, bb: b.bb}
                }, pa(a))
            }

            function Jj(a, c, b) {
                void 0 === b && (b = !1);
                return new I(function (d, e) {
                    function f(k, l) {
                        l();
                        d()
                    }

                    var g = a.slice();
                    g.push({Z: f, Ea: f});
                    var h = jc(g, function (k, l) {
                        var m = b ? k.Z : k.Ea;
                        if (m) try {
                            m(c, l)
                        } catch (p) {
                            h(Bq), e(p)
                        } else l()
                    });
                    h(Oj)
                })
            }

            function Kb(a, c, b) {
                var d = b || "as";
                if (a.postMessage && !a.attachEvent) {
                    b = ia(a);
                    var e = "__ym__promise_" + Va(a) +
                        "_" + Va(a), f = B;
                    d = D(a, d, function (g) {
                        try {
                            var h = g.data
                        } catch (k) {
                            return
                        }
                        h === e && (f(), g.stopPropagation && g.stopPropagation(), c())
                    });
                    f = b.D(a, ["message"], d);
                    a.postMessage(e, "*")
                } else V(a, c, 0, d)
            }

            function rh(a, c, b, d, e) {
                void 0 === d && (d = 1);
                void 0 === e && (e = "itc");
                c = jc(c, b);
                kc(a, c, d)(Sa(D(a, e), B))
            }

            function kc(a, c, b, d) {
                void 0 === b && (b = 1);
                void 0 === d && (d = Pj);
                xg = Infinity === b;
                return za(function (e, f) {
                    function g() {
                        try {
                            var k = c(d(a, b));
                            h = h.concat(k)
                        } catch (l) {
                            return e(l)
                        }
                        c(Cq);
                        if (c(Jd)) return f(h), Qj(a);
                        xg ? (c(d(a, 1E4)), f(h),
                            Qj(a)) : V(a, g, 100)
                    }

                    var h = [];
                    Dq(g)
                })
            }

            function Qj(a) {
                if (yg.length) {
                    var c = yg.shift();
                    xg ? c() : V(a, c, 100)
                } else zg = !1
            }

            function Dq(a) {
                zg ? yg.push(a) : (zg = !0, a())
            }

            function yf(a) {
                return za(function (c, b) {
                    b(a)
                })
            }

            function Jo(a) {
                return za(function (c, b) {
                    a.then(b, c)
                })
            }

            function Eq(a) {
                var c = [], b = 0;
                return za(function (d, e) {
                    z(function (f, g) {
                        f(Sa(d, function (h) {
                            try {
                                c[g] = h, b += 1, b === a.length && e(c)
                            } catch (k) {
                                d(k)
                            }
                        }))
                    }, a)
                })
            }

            function Io(a) {
                var c = [], b = !1;
                return za(function (d, e) {
                    function f(g) {
                        c.push(g) === a.length && d(c)
                    }

                    z(function (g) {
                        g(Sa(f,
                            function (h) {
                                if (!b) try {
                                    e(h), b = !0
                                } catch (k) {
                                    f(k)
                                }
                            }))
                    }, a)
                })
            }

            function Sa(a, c) {
                return function (b) {
                    return b(a, c)
                }
            }

            function jc(a, c) {
                void 0 === c && (c = O);
                return za({ab: a, de: c, Oe: !1, ya: 0})
            }

            function Oj(a) {
                function c() {
                    function d() {
                        b = !0;
                        a.ya += 1
                    }

                    b = !1;
                    a.de(a.ab[a.ya], function () {
                        d()
                    });
                    b || (a.ya += 1, d = u(a, Oj))
                }

                for (var b = !0; !Jd(a) && b;) c()
            }

            function Pj(a, c) {
                return function (b) {
                    var d = fa(a), e = d(aa);
                    return Rj(function (f, g) {
                        d(aa) - e >= c && g(Sj)
                    })(b)
                }
            }

            function Le(a, c) {
                return function (b) {
                    var d = fa(a), e = d(aa);
                    return Me(function (f) {
                        d(aa) -
                        e >= c && Sj(f)
                    })(b)
                }
            }

            function Me(a) {
                return function (c) {
                    for (var b; c.ab.length && !Jd(c);) b = c.ab.pop(), b = c.de(b, c.ab), a(c);
                    return b
                }
            }

            function Fq(a) {
                Jd(a) && Xa(ic("i"));
                var c = a.de(a.ab[a.ya]);
                a.ya += 1;
                return c
            }

            function Cq(a) {
                a.Oe = !1
            }

            function Sj(a) {
                a.Oe = !0
            }

            function Bq(a) {
                a.ya = a.ab.length
            }

            function Jd(a) {
                return a.Oe || a.ab.length <= a.ya
            }

            function ub(a) {
                a = fa(a);
                return Math.round(a(Ag) / 50)
            }

            function Ag(a) {
                var c = a.Ba, b = c[1];
                a = c[0] && b ? b() : aa(a) - a.Di;
                return Math.round(a)
            }

            function jg(a) {
                return Math.round(aa(a) / 1E3)
            }

            function lb(a) {
                return Math.floor(aa(a) /
                    1E3 / 60)
            }

            function aa(a) {
                var c = a.We;
                return 0 !== c ? c : Bg(a.l, a.Ba)
            }

            function fg(a) {
                var c = ia(a), b = Tj(a), d = {l: a, We: 0, Ba: b, Di: Bg(a, b)}, e = b[1];
                b[0] && e || c.D(a, ["beforeunload", "unload"], function () {
                    0 === d.We && (d.We = Bg(a, d.Ba))
                });
                return za(d)
            }

            function Gq(a) {
                return (10 > a ? "0" : "") + a
            }

            function Ci(a, c, b) {
                function d() {
                    f = 0;
                    g && (g = !1, f = V(a, d, b), e.O(h))
                }

                var e = wd(a), f, g = !1, h;
                c.D(function (k) {
                    g = !0;
                    h = k;
                    f || d();
                    return B
                });
                return e
            }

            function Hq(a, c) {
                return a.clearInterval(c)
            }

            function Iq(a, c, b, d) {
                return a.setInterval(D(a, "i.err." + (d ||
                    "def"), c), b)
            }

            function V(a, c, b, d) {
                return Be(a, D(a, "d.err." + (d || "def"), c), b)
            }

            function id(a) {
                var c = {};
                return {
                    D: function (b, d) {
                        z(function (e) {
                            n(c, e) || (c[e] = wd(a));
                            c[e].D(d)
                        }, b);
                        return this
                    }, oa: function (b, d) {
                        z(function (e) {
                            n(c, e) && c[e].oa(d)
                        }, b);
                        return this
                    }, O: function (b, d) {
                        return n(c, b) ? D(a, "e." + d, c[b].O)(d) : []
                    }
                }
            }

            function wd(a) {
                var c = [], b = {};
                b.Ek = c;
                b.D = t(qa("push", c), u(b, O));
                b.oa = t(Ib(Pb(a))(c), Ib(qa("splice", c))(1), u(b, O));
                b.O = t(O, Ib(ha), Jq(c));
                return b
            }

            function Uj(a, c, b, d, e) {
                var f = a.addEventListener &&
                    a.removeEventListener, g = !f && a.attachEvent && a.detachEvent;
                if (f || g) if (e = e ? f ? "removeEventListener" : "detachEvent" : f ? "addEventListener" : "attachEvent", f) a[e](c, b, d); else a[e]("on" + c, b)
            }

            function C(a, c, b) {
                return function () {
                    return D(arguments[0], a, c, b).apply(this, arguments)
                }
            }

            function D(a, c, b, d, e) {
                var f = Xa, g = b || f;
                return function () {
                    var h = d;
                    try {
                        h = g.apply(e || null, arguments)
                    } catch (k) {
                        oe(a, c, k)
                    }
                    return h
                }
            }

            function Bg(a, c) {
                var b = c || Tj(a), d = b[0];
                b = b[1];
                return !isNaN(d) && T(b) ? Math.round(b() + d) : a.Date.now ? a.Date.now() :
                    (new a.Date).getTime()
            }

            function Tj(a) {
                a = Rf(a);
                var c = n(a, "timing.navigationStart"), b = n(a, "now");
                b && (b = F(b, a));
                return [c, b]
            }

            function Rf(a) {
                return n(a, "performance") || n(a, "webkitPerformance")
            }

            function oe(a, c, b) {
                var d = "u.a.e", e = "";
                b && ("object" === typeof b ? (b.unk && Xa(b), d = b.message, e = "string" === typeof b.stack && b.stack.replace(/\n/g, "\\n") || "n.s.e.s") : d = "" + b);
                Kq(d) || Ja(t(qa("indexOf", d), ka(-1), Tb), Lq) || Mq(d) && .1 <= a.Math.random() || z(t(O, Oc(["jserrs", d, c, e]), ha), Vj)
            }

            function ff() {
                var a = Ca(arguments);
                return Xa(Ta(a))
            }

            function Ta(a) {
                var c = "";
                ca(a) ? c = J(".", a) : wa(a) && (c = a);
                return ic("err.kn(" + sa.jb + ")" + c)
            }

            function Nq(a) {
                this.message = a
            }

            function ib(a, c) {
                var b = c.da;
                if (b) {
                    var d = b.split(":");
                    b = d[1];
                    d = Wj(Fh(d[0]));
                    if ("1" === b || d) return
                }
                b = Oq(a);
                1E3 === b.length && b.shift();
                b.push(c)
            }

            function Fl(a, c, b, d, e) {
                var f = "object" === typeof a ? a : {id: a, ba: d, wc: e, ea: b};
                a = M(function (g, h) {
                    var k = h[1], l = k.bb;
                    k = f[k.ia];
                    g[h[0]] = l ? l(k) : k;
                    return g
                }, {}, pa(c));
                bj(a, a.ea || {});
                return a
            }

            function Pq(a, c) {
                return M(function (b, d) {
                        b[c[d[0]].ia] = d[1];
                        return b
                    },
                    {}, pa(a))
            }

            function Qq(a) {
                a = N(a);
                return gc[a] && gc[a].Nj || null
            }

            function Xj(a) {
                a = N(a);
                return gc[a] && gc[a].Mj
            }

            function bj(a, c) {
                var b = N(a), d = n(c, "__ym.turbo_page"), e = n(c, "__ym.turbo_page_id");
                gc[b] || (gc[b] = {});
                if (d || e) gc[b].Mj = d, gc[b].Nj = e
            }

            function Yj(a) {
                return Ne(a) || od(a) || /mobile/i.test(gb(a)) || !W(n(a, "orientation"))
            }

            function pf(a, c) {
                if (Kd(a) && c) {
                    var b = gb(a).match(Cg);
                    if (b && b.length) return +b[1] >= c
                }
                return !1
            }

            function qf(a, c) {
                var b = gb(a);
                return b && (b = b.match(Rq)) && 1 < b.length ? Ga(b[1]) >= c : !1
            }

            function vg(a) {
                return K("prerender",
                    A(u(n(a, "document"), n), ["webkitVisibilityState", "visibilityState"]))
            }

            function Va(a, c, b) {
                var d = W(b);
                W(c) && d ? (d = 1, c = 1073741824) : d ? d = 1 : (d = c, c = b);
                return a.Math.floor(a.Math.random() * (c - d)) + d
            }

            function Sq() {
                var a = Ca(arguments), c = a[0];
                for (a = a.slice(1); a.length;) {
                    var b = a.shift(), d;
                    for (d in b) Ad(b, d) && (c[d] = b[d]);
                    Ad(b, "toString") && (c.toString = b.toString)
                }
                return c
            }

            function Zj(a) {
                return W(a) ? [] : Ld(function (c, b) {
                    c.push([b, a[b]]);
                    return c
                }, [], ak(a))
            }

            function ak(a) {
                var c = [], b;
                for (b in a) Ad(a, b) && c.push(b);
                return c
            }

            function Fh(a) {
                try {
                    return parseInt(a, 10)
                } catch (c) {
                    return null
                }
            }

            function qe(a, c) {
                return a.isFinite(c) && !a.isNaN(c) && "[object Number]" === Dg(c)
            }

            function Tq(a) {
                for (var c = [], b = a.length - 1; 0 <= b; --b) c[a.length - 1 - b] = a[b];
                return c
            }

            function Oa(a, c) {
                z(t(O, qa("push", a)), c);
                return a
            }

            function Eg(a, c) {
                return Array.prototype.sort.call(c, a)
            }

            function Ed(a) {
                return a.splice(0, a.length)
            }

            function ya(a) {
                return a ? ca(a) ? a : Oe ? Oe(a) : "number" === typeof a.length && 0 <= a.length ? bk(a) : [] : []
            }

            function Pe(a, c, b) {
                return b ? a : c
            }

            function Uq(a,
                        c) {
                return Ld(function (b, d, e) {
                    d = a(d, e);
                    return b.concat(ca(d) ? d : [d])
                }, [], c)
            }

            function ck(a, c) {
                return Ld(function (b, d, e) {
                    b.push(a(d, e));
                    return b
                }, [], c)
            }

            function Vq(a, c) {
                if (!Kd(a)) return !0;
                try {
                    c.call({0: !0, length: -Math.pow(2, 32) + 1}, function () {
                        throw 1;
                    })
                } catch (b) {
                    return !1
                }
                return !0
            }

            function ca(a) {
                if (Md) return Md(a);
                (Md = Ma(Array.isArray, "isArray")) || (Md = Wq);
                return Md(a)
            }

            function Xq(a, c) {
                for (var b = "", d = 0; d < c.length; d += 1) b += "" + (d ? a : "") + c[d];
                return b
            }

            function Yq(a, c) {
                return 1 <= dk(ka(a), c).length
            }

            function Zq(a,
                        c) {
                for (var b = 0; b < c.length; b += 1) if (a.call(c, c[b], b)) return c[b]
            }

            function dk(a, c) {
                return Ld(function (b, d, e) {
                    a(d, e) && b.push(d);
                    return b
                }, [], c)
            }

            function ag(a, c, b) {
                try {
                    if (T(c)) {
                        var d = Ca(arguments).slice(3);
                        ma(b) ? c.apply(void 0, d) : F.apply(void 0, xa([c, b], d))()
                    }
                } catch (e) {
                    Be(a, u(e, Xa), 0)
                }
            }

            function Xa(a) {
                throw a;
            }

            function Be(a, c, b) {
                return ec("setTimeout", a)(c, b)
            }

            function na(a, c) {
                return ec("clearTimeout", a)(c)
            }

            function vd() {
                return []
            }

            function Ac() {
                return {}
            }

            function ec(a, c) {
                var b = n(c, a), d = n(c, "constructor.prototype." +
                    a) || b;
                try {
                    if (d && d.apply) return function () {
                        return d.apply(c, arguments)
                    }
                } catch (e) {
                    return b
                }
                return d
            }

            function pb(a, c, b) {
                return function () {
                    var d = Ca(arguments), e = d[0];
                    d = d.slice(1);
                    var f = H(e), g = b ? "global" : "m1060", h = f.o(g, {}), k = n(h, a);
                    k || (k = x(c), h[a] = k, f.C(g, h));
                    return k.apply(void 0, xa([e], d))
                }
            }

            function Gc(a, c) {
                void 0 === c && (c = {});
                if (!a || 1 > a.length) return c;
                M(function (b, d, e) {
                    if (e === a.length - 1) return b;
                    e === a.length - 2 ? b[d] = a[e + 1] : b[d] || (b[d] = {});
                    return b[d]
                }, c, a);
                return c
            }

            function n(a, c) {
                return a ? M(function (b,
                                       d) {
                    if (ma(b)) return b;
                    try {
                        return b[d]
                    } catch (e) {
                    }
                    return null
                }, a, c.split(".")) : null
            }

            function Nd(a) {
                a = a.Ya = a.Ya || {};
                var c = a._metrika = a._metrika || {};
                return {
                    Va: function (b, d) {
                        Fg.call(c, b) || (c[b] = d);
                        return this
                    }, C: function (b, d) {
                        c[b] = d;
                        return this
                    }, o: function (b, d) {
                        var e = c[b];
                        return Fg.call(c, b) || W(d) ? e : d
                    }
                }
            }

            function Ad(a, c) {
                return ma(a) ? !1 : Fg.call(a, c)
            }

            function x(a, c) {
                var b = [], d = [];
                var e = c ? c : O;
                return function () {
                    var f = Ca(arguments), g = e.apply(void 0, f), h = ek(g, d);
                    if (-1 !== h) return b[h];
                    f = a.apply(void 0, f);
                    b.push(f);
                    d.push(g);
                    return f
                }
            }

            function Pb(a) {
                if (Gg) return Gg;
                var c = !1;
                try {
                    c = [].indexOf && 0 === [void 0].indexOf(void 0)
                } catch (d) {
                }
                var b = a.Array && a.Array.prototype && Ma(a.Array.prototype.indexOf, "indexOf");
                return Gg = a = c && b ? function (d, e) {
                    return b.call(e, d)
                } : $q
            }

            function $q(a, c) {
                for (var b = 0; b < c.length; b += 1) if (c[b] === a) return b;
                return -1
            }

            function ha(a, c) {
                return c ? a(c) : a()
            }

            function t() {
                var a = Ca(arguments), c = a.shift();
                return function () {
                    var b = c.apply(void 0, arguments);
                    return M(fk, b, a)
                }
            }

            function xd(a, c) {
                return E([a, c], M)
            }

            function Ld(a, c, b) {
                for (var d = 0, e = b.length; d < e;) c = a(c, b[d], d), d += 1;
                return c
            }

            function Ya(a) {
                return qa("test", a)
            }

            function qa(a, c) {
                return F(c[a], c)
            }

            function u(a, c) {
                return F(c, null, a)
            }

            function E(a, c) {
                return F.apply(void 0, xa([c, null], a))
            }

            function ar() {
                var a = Ca(arguments), c = a[0], b = a[1], d = a.slice(2);
                return function () {
                    var e = xa(d, Ca(arguments));
                    if (Function.prototype.call) return Function.prototype.call.apply(c, xa([b], e));
                    if (b) {
                        for (var f = "_b"; b[f];) f += "_" + f.length;
                        b[f] = c;
                        e = b[f] && gk(f, e, b);
                        delete b[f];
                        return e
                    }
                    return gk(c,
                        e)
                }
            }

            function gk(a, c, b) {
                void 0 === c && (c = []);
                b = b || {};
                var d = c.length, e = a;
                T(e) && (e = "d", b[e] = a);
                var f;
                d ? 1 === d ? f = b[e](c[0]) : 2 === d ? f = b[e](c[0], c[1]) : 3 === d ? f = b[e](c[0], c[1], c[2]) : 4 === d && (f = b[e](c[0], c[1], c[2], c[3])) : f = b[e]();
                return f
            }

            function Ca(a) {
                if (Oe) try {
                    return Oe(a)
                } catch (c) {
                }
                return bk(a)
            }

            function bk(a) {
                for (var c = a.length, b = [], d = 0; d < c; d += 1) b.push(a[d]);
                return b
            }

            function La(a) {
                return !Ua(a) && !W(a) && "[object Object]" === Dg(a)
            }

            function ma(a) {
                return W(a) || Ua(a)
            }

            function T(a) {
                return "function" === typeof a
            }

            function Ib(a) {
                return function (c) {
                    return function (b) {
                        return a(b,
                            c)
                    }
                }
            }

            function la(a) {
                return function (c) {
                    return function (b) {
                        return a(c, b)
                    }
                }
            }

            function fk(a, c) {
                return c(a)
            }

            function Gp(a) {
                return a.replace(/\^/g, "\\^").replace(/\$/g, "\\$").replace(dg, "\\.").replace(/\[/g, "\\[").replace(/\]/g, "\\]").replace(/\|/g, "\\|").replace(/\(/g, "\\(").replace(/\)/g, "\\)").replace(/\?/g, "\\?").replace(/\*/g, "\\*").replace(/\+/g, "\\+").replace(/\{/g, "\\{").replace(/\}/g, "\\}")
            }

            function br(a) {
                return "" + a
            }

            function pc(a, c) {
                return !(!a || -1 === he(a, c))
            }

            function he(a, c) {
                if (hk) var b = hk.call(a,
                    c); else a:{
                    b = 0;
                    for (var d = a.length - c.length, e = 0; e < a.length; e += 1) {
                        b = a[e] === c[b] ? b + 1 : 0;
                        if (b === c.length) {
                            b = e - c.length + 1;
                            break a
                        }
                        if (!b && e > d) break
                    }
                    b = -1
                }
                return b
            }

            function wa(a) {
                return "string" === typeof a
            }

            function Dg(a) {
                return Object.prototype.toString.call(a)
            }

            function Hg(a, c) {
                Hg = Object.setPrototypeOf || {__proto__: []} instanceof Array && function (b, d) {
                    b.__proto__ = d
                } || function (b, d) {
                    for (var e in d) d.hasOwnProperty(e) && (b[e] = d[e])
                };
                return Hg(a, c)
            }

            function Ma(a, c) {
                return Aa(c, a) && a
            }

            function Aa(a, c) {
                var b = Qe(a, c);
                c && !b && Ig.push([a, c]);
                return b
            }

            function Qe(a, c) {
                if (!c || "function" !== typeof c) return !1;
                try {
                    var b = "" + c
                } catch (h) {
                    return !1
                }
                var d = b.length;
                if (d > 35 + a.length) return !1;
                for (var e = d - 13, f = 0, g = 8; g < d; g += 1) {
                    f = "[native code]"[f] === b[g] || 7 === f && "-" === b[g] ? f + 1 : 0;
                    if (12 === f) return !0;
                    if (!f && g > e) break
                }
                return !1
            }

            function B() {
            }

            function Tb(a) {
                return !a
            }

            function yb(a, c) {
                return c
            }

            function O(a) {
                return a
            }

            function Ka(a, c) {
                function b() {
                    this.constructor = a
                }

                Hg(a, c);
                a.prototype = null === c ? Object.create(c) : (b.prototype = c.prototype, new b)
            }

            function xa() {
                for (var a = 0, c = 0, b = arguments.length; c < b; c++) a += arguments[c].length;
                a = Array(a);
                var d = 0;
                for (c = 0; c < b; c++) for (var e = arguments[c], f = 0, g = e.length; f < g; f++, d++) a[d] = e[f];
                return a
            }

            function cr() {
            }

            function dr(a, c) {
                return function () {
                    a.apply(c, arguments)
                }
            }

            function Fa(a) {
                if (!(this instanceof Fa)) throw new TypeError("Promises must be constructed via new");
                if ("function" !== typeof a) throw new TypeError("not a function");
                this.Na = 0;
                this.af = !1;
                this.Xa = void 0;
                this.Lb = [];
                ik(a, this)
            }

            function jk(a, c) {
                for (; 3 ===
                       a.Na;) a = a.Xa;
                0 === a.Na ? a.Lb.push(c) : (a.af = !0, Fa.bf(function () {
                    var b = 1 === a.Na ? c.cj : c.gj;
                    if (null === b) (1 === a.Na ? Jg : Od)(c.gb, a.Xa); else {
                        try {
                            var d = b(a.Xa)
                        } catch (e) {
                            Od(c.gb, e);
                            return
                        }
                        Jg(c.gb, d)
                    }
                }))
            }

            function Jg(a, c) {
                try {
                    if (c === a) throw new TypeError("A promise cannot be resolved with itself.");
                    if (c && ("object" === typeof c || "function" === typeof c)) {
                        var b = c.then;
                        if (c instanceof Fa) {
                            a.Na = 3;
                            a.Xa = c;
                            Kg(a);
                            return
                        }
                        if ("function" === typeof b) {
                            ik(dr(b, c), a);
                            return
                        }
                    }
                    a.Na = 1;
                    a.Xa = c;
                    Kg(a)
                } catch (d) {
                    Od(a, d)
                }
            }

            function Od(a, c) {
                a.Na =
                    2;
                a.Xa = c;
                Kg(a)
            }

            function Kg(a) {
                2 === a.Na && 0 === a.Lb.length && Fa.bf(function () {
                    a.af || Fa.dh(a.Xa)
                });
                for (var c = 0, b = a.Lb.length; c < b; c++) jk(a, a.Lb[c]);
                a.Lb = null
            }

            function er(a, c, b) {
                this.cj = "function" === typeof a ? a : null;
                this.gj = "function" === typeof c ? c : null;
                this.gb = b
            }

            function ik(a, c) {
                var b = !1;
                try {
                    a(function (d) {
                        b || (b = !0, Jg(c, d))
                    }, function (d) {
                        b || (b = !0, Od(c, d))
                    })
                } catch (d) {
                    b || (b = !0, Od(c, d))
                }
            }

            function ln(a) {
                for (var c = a.length, b = 0, d = 255, e = 255, f, g, h; c;) {
                    f = 21 < c ? 21 : c;
                    c -= f;
                    do g = "string" === typeof a ? a.charCodeAt(b) : a[b],
                        b += 1, 255 < g && (h = g >> 8, g &= 255, g ^= h), d += g, e += d; while (--f);
                    d = (d & 255) + (d >> 8);
                    e = (e & 255) + (e >> 8)
                }
                a = (d & 255) + (d >> 8) << 8 | (e & 255) + (e >> 8);
                return 65535 === a ? 0 : a
            }

            function tb(a, c) {
                if (!c) return null;
                try {
                    return a.JSON.parse(c)
                } catch (b) {
                    return null
                }
            }

            function oc(a) {
                a = "" + a;
                for (var c = 2166136261, b = a.length, d = 0; d < b; d += 1) c ^= a.charCodeAt(d), c += (c << 1) + (c << 4) + (c << 7) + (c << 8) + (c << 24);
                return c >>> 0
            }

            function sj(a, c, b, d, e) {
                void 0 === e && (e = !1);
                return kg(a, c, "", -100, b, d, e)
            }

            function Dc(a, c, b) {
                void 0 === c && (c = "_ym_");
                void 0 === b && (b = "");
                var d =
                    fr(a), e = 1 === (d || "").split(".").length ? d : "." + d, f = b ? "_" + b : "";
                return {
                    Rb: function (g, h, k) {
                        sj(a, "" + c + g + f, h || e, k);
                        return this
                    }, o: function (g) {
                        return je(a, "" + c + g + f)
                    }, C: function (g, h, k, l, m) {
                        kg(a, "" + c + g + f, h, k, l || e, m);
                        return this
                    }
                }
            }

            function Gl(a, c, b, d) {
                var e = kk[b];
                return e ? function () {
                    var f = Ca(arguments);
                    try {
                        var g = d.apply(void 0, f);
                        var h = H(a);
                        h.Va("mt", {});
                        var k = h.o("mt"), l = k[e];
                        k[e] = l ? l + 1 : 1
                    } catch (m) {
                        Xa(m)
                    }
                    return g
                } : d
            }

            function Hc(a, c) {
                var b = gr(a);
                return b ? (b.href = c, {
                    protocol: b.protocol,
                    host: b.host,
                    port: b.port,
                    hostname: b.hostname,
                    hash: b.hash,
                    search: b.search,
                    query: b.search.replace(/^\?/, ""),
                    pathname: b.pathname || "/",
                    path: (b.pathname || "/") + b.search,
                    href: b.href
                }) : {}
            }

            function lk(a) {
                return (a = S(a).hash.split("#")[1]) ? a.split("?")[0] : ""
            }

            function hr(a, c) {
                var b = lk(a);
                mk = Iq(a, function () {
                    var d = lk(a);
                    d !== b && (c(), b = d)
                }, 200, "t.h");
                return F(Hq, null, a, mk)
            }

            function ir(a, c, b) {
                var d, e, f = c.ba, g = c.Ye, h = c.Hc, k = H(a), l = Da((d = {}, d.wh = 1, d.pv = 1, d));
                Yd(f) && a.Ya && a.Ya.Direct && l.C("ad", "1");
                g && l.C("ut", "1");
                f = k.o("lastReferrer");
                d = S(a).href;
                h = {G: (e = {}, e["page-url"] = h || d, e["page-ref"] = f, e), H: l};
                b(h, c)["catch"](D(a, "g.s"));
                k.C("lastReferrer", d)
            }

            function jr(a, c, b) {
                function d() {
                    na(a, h);
                    g(!1)
                }

                function e() {
                    k = !0;
                    g(!1);
                    c()
                }

                function f() {
                    na(a, h);
                    if (k) g(!1); else {
                        var Y = Math.max(0, b - (q ? r : r + l(aa) - v));
                        Y ? h = V(a, e, Y, "u.t.d.c") : e()
                    }
                }

                function g(Y) {
                    z(function (Q) {
                        var oa = Q[0], ta = Q[1];
                        Q = Q[2];
                        Y ? w.D(oa, ta, Q) : w.kc(oa, ta, Q)
                    }, G)
                }

                var h = 0, k = !1;
                if (sg(a)) return h = V(a, c, b, "u.t.d"), d;
                var l = fa(a), m = !1, p = !1, q = !0, r = 0, v = l(aa), w = ia(a), G = [[a, ["blur"], function () {
                    q = m = p = !0;
                    r +=
                        l(aa) - v;
                    v = l(aa);
                    f()
                }], [a, ["focus"], function () {
                    m || p || (r = 0);
                    v = l(aa);
                    m = p = !0;
                    q = !1;
                    f()
                }], [a.document, ["click", "mousemove", "keydown", "scroll"], function () {
                    p || (m = !0, q = !1, p = !0, f())
                }]];
                g(!0);
                f();
                return d
            }

            function ef(a, c, b, d) {
                return function () {
                    if (Ha(a, c)) {
                        var e = Ca(arguments);
                        return d.apply(void 0, e)
                    }
                }
            }

            function qb(a, c) {
                a = [a[0] >>> 16, a[0] & 65535, a[1] >>> 16, a[1] & 65535];
                c = [c[0] >>> 16, c[0] & 65535, c[1] >>> 16, c[1] & 65535];
                var b = [0, 0, 0, 0];
                b[3] += a[3] * c[3];
                b[2] += b[3] >>> 16;
                b[3] &= 65535;
                b[2] += a[2] * c[3];
                b[1] += b[2] >>> 16;
                b[2] &= 65535;
                b[2] += a[3] * c[2];
                b[1] += b[2] >>> 16;
                b[2] &= 65535;
                b[1] += a[1] * c[3];
                b[0] += b[1] >>> 16;
                b[1] &= 65535;
                b[1] += a[2] * c[2];
                b[0] += b[1] >>> 16;
                b[1] &= 65535;
                b[1] += a[3] * c[1];
                b[0] += b[1] >>> 16;
                b[1] &= 65535;
                b[0] += a[0] * c[3] + a[1] * c[2] + a[2] * c[1] + a[3] * c[0];
                b[0] &= 65535;
                return [b[0] << 16 | b[1], b[2] << 16 | b[3]]
            }

            function hc(a, c) {
                a = [a[0] >>> 16, a[0] & 65535, a[1] >>> 16, a[1] & 65535];
                c = [c[0] >>> 16, c[0] & 65535, c[1] >>> 16, c[1] & 65535];
                var b = [0, 0, 0, 0];
                b[3] += a[3] + c[3];
                b[2] += b[3] >>> 16;
                b[3] &= 65535;
                b[2] += a[2] + c[2];
                b[1] += b[2] >>> 16;
                b[2] &= 65535;
                b[1] += a[1] + c[1];
                b[0] +=
                    b[1] >>> 16;
                b[1] &= 65535;
                b[0] += a[0] + c[0];
                b[0] &= 65535;
                return [b[0] << 16 | b[1], b[2] << 16 | b[3]]
            }

            function Rc(a, c) {
                c %= 64;
                if (32 === c) return [a[1], a[0]];
                if (32 > c) return [a[0] << c | a[1] >>> 32 - c, a[1] << c | a[0] >>> 32 - c];
                c -= 32;
                return [a[1] << c | a[0] >>> 32 - c, a[0] << c | a[1] >>> 32 - c]
            }

            function jb(a, c) {
                c %= 64;
                return 0 === c ? a : 32 > c ? [a[0] << c | a[1] >>> 32 - c, a[1] << c] : [a[1] << c - 32, 0]
            }

            function ua(a, c) {
                return [a[0] ^ c[0], a[1] ^ c[1]]
            }

            function nk(a) {
                a = ua(a, [0, a[0] >>> 1]);
                a = qb(a, [4283543511, 3981806797]);
                a = ua(a, [0, a[0] >>> 1]);
                a = qb(a, [3301882366, 444984403]);
                return a =
                    ua(a, [0, a[0] >>> 1])
            }

            function kr(a, c) {
                void 0 === c && (c = 210);
                var b = a || "", d = c || 0, e = b.length - b.length % 16;
                d = {R: [0, d], T: [0, d]};
                for (var f = 0; f < e; f += 16) {
                    var g = d,
                        h = [a.charCodeAt(f + 4) & 255 | (a.charCodeAt(f + 5) & 255) << 8 | (a.charCodeAt(f + 6) & 255) << 16 | (a.charCodeAt(f + 7) & 255) << 24, a.charCodeAt(f) & 255 | (a.charCodeAt(f + 1) & 255) << 8 | (a.charCodeAt(f + 2) & 255) << 16 | (a.charCodeAt(f + 3) & 255) << 24],
                        k = [a.charCodeAt(f + 12) & 255 | (a.charCodeAt(f + 13) & 255) << 8 | (a.charCodeAt(f + 14) & 255) << 16 | (a.charCodeAt(f + 15) & 255) << 24, a.charCodeAt(f + 8) & 255 | (a.charCodeAt(f +
                            9) & 255) << 8 | (a.charCodeAt(f + 10) & 255) << 16 | (a.charCodeAt(f + 11) & 255) << 24];
                    h = qb(h, Re);
                    h = Rc(h, 31);
                    h = qb(h, Se);
                    g.R = ua(g.R, h);
                    g.R = Rc(g.R, 27);
                    g.R = hc(g.R, g.T);
                    g.R = hc(qb(g.R, [0, 5]), [0, 1390208809]);
                    k = qb(k, Se);
                    k = Rc(k, 33);
                    k = qb(k, Re);
                    g.T = ua(g.T, k);
                    g.T = Rc(g.T, 31);
                    g.T = hc(g.T, g.R);
                    g.T = hc(qb(g.T, [0, 5]), [0, 944331445])
                }
                e = b.length % 16;
                f = b.length - e;
                g = [0, 0];
                h = [0, 0];
                switch (e) {
                    case 15:
                        h = ua(h, jb([0, b.charCodeAt(f + 14)], 48));
                    case 14:
                        h = ua(h, jb([0, b.charCodeAt(f + 13)], 40));
                    case 13:
                        h = ua(h, jb([0, b.charCodeAt(f + 12)], 32));
                    case 12:
                        h = ua(h,
                            jb([0, b.charCodeAt(f + 11)], 24));
                    case 11:
                        h = ua(h, jb([0, b.charCodeAt(f + 10)], 16));
                    case 10:
                        h = ua(h, jb([0, b.charCodeAt(f + 9)], 8));
                    case 9:
                        h = ua(h, [0, b.charCodeAt(f + 8)]), h = qb(h, Se), h = Rc(h, 33), h = qb(h, Re), d.T = ua(d.T, h);
                    case 8:
                        g = ua(g, jb([0, b.charCodeAt(f + 7)], 56));
                    case 7:
                        g = ua(g, jb([0, b.charCodeAt(f + 6)], 48));
                    case 6:
                        g = ua(g, jb([0, b.charCodeAt(f + 5)], 40));
                    case 5:
                        g = ua(g, jb([0, b.charCodeAt(f + 4)], 32));
                    case 4:
                        g = ua(g, jb([0, b.charCodeAt(f + 3)], 24));
                    case 3:
                        g = ua(g, jb([0, b.charCodeAt(f + 2)], 16));
                    case 2:
                        g = ua(g, jb([0, b.charCodeAt(f +
                            1)], 8));
                    case 1:
                        g = ua(g, [0, b.charCodeAt(f)]), g = qb(g, Re), g = Rc(g, 31), g = qb(g, Se), d.R = ua(d.R, g)
                }
                d.R = ua(d.R, [0, b.length]);
                d.T = ua(d.T, [0, b.length]);
                d.R = hc(d.R, d.T);
                d.T = hc(d.T, d.R);
                d.R = nk(d.R);
                d.T = nk(d.T);
                d.R = hc(d.R, d.T);
                d.T = hc(d.T, d.R);
                return ("00000000" + (d.R[0] >>> 0).toString(16)).slice(-8) + ("00000000" + (d.R[1] >>> 0).toString(16)).slice(-8) + ("00000000" + (d.T[0] >>> 0).toString(16)).slice(-8) + ("00000000" + (d.T[1] >>> 0).toString(16)).slice(-8)
            }

            function Pd(a, c, b) {
                var d = c.getAttribute("itemtype");
                b = xb('[itemprop~="' +
                    b + '"]', c);
                return d ? Z(function (e) {
                    return e.parentNode && dc("[itemtype]", a, e.parentNode) === c
                }, b) : b
            }

            function cb(a, c, b) {
                return (a = Pd(a, c, b)) && a.length ? a[0] : null
            }

            function Za(a) {
                if (!a) return "";
                a = ca(a) ? a : [a];
                return a.length ? a[0].getAttribute("content") || Sb(a[0]) : ""
            }

            function ok(a) {
                return a ? a.attributes && a.getAttribute("datetime") ? a.getAttribute("datetime") : Za(a) : ""
            }

            function ld(a, c, b) {
                a = c && (pc(c.className, "ym-disable-keys") || pc(c.className, "-metrika-nokeys"));
                return b && c ? a || !!sq(["ym-disable-keys", "-metrika-nokeys"],
                    c).length : a
            }

            function Bf(a, c) {
                return He(c) ? "password" === c.type || c.name && K(c.name.toLowerCase(), pk) || c.id && K(c.id.toLowerCase(), pk) : !1
            }

            function qk(a, c) {
                var b = Math.max(0, Math.min(c, 65535));
                Oa(a, [b >> 8, b & 255])
            }

            function Mb(a, c) {
                Oa(a, [c & 255])
            }

            function fb(a, c, b) {
                return -1 !== Pb(a)(b, lr) ? (Mb(c, b), !1) : !0
            }

            function R(a, c) {
                for (var b = Math.max(0, c | 0); 127 < b;) Oa(a, [b & 127 | 128]), b >>= 7;
                Oa(a, [b])
            }

            function Lg(a, c) {
                R(a, c.length);
                for (var b = 0; b < c.length; b += 1) R(a, c.charCodeAt(b))
            }

            function Mg(a, c) {
                var b = c;
                255 < b.length && (b = b.substr(0,
                    255));
                a.push(b.length);
                for (var d = 0; d < b.length; d += 1) qk(a, b.charCodeAt(d))
            }

            function mr(a, c) {
                var b = [];
                if (fb(a, b, 27)) return [];
                R(b, c);
                return b
            }

            function nr(a, c) {
                var b = Ia(c);
                if (!b) return c[Wa] = -1, null;
                var d = +c[Wa];
                if (!isFinite(d) || 0 >= d) return null;
                if (c.attributes) for (var e = c; e;) {
                    if (e.attributes.ik) return null;
                    e = e.parentElement
                }
                e = 64;
                var f = Fe(a, c), g = f && f[Wa] ? f[Wa] : 0;
                0 > g && (g = 0);
                b = (b || "").toUpperCase();
                var h = or()[b];
                h || (e |= 2);
                var k = Aj(a, c);
                k || (e |= 4);
                var l = Ff(a, c);
                (f = f ? Ff(a, f) : null) && l[0] === f[0] && l[1] === f[1] &&
                l[2] === f[2] && l[3] === f[3] && (e |= 8);
                sc[d].$f = l[0] + "x" + l[1];
                sc[d].size = l[2] + "x" + l[3];
                c.id && "string" === typeof c.id && (e |= 32);
                f = [];
                if (fb(a, f, 1)) return null;
                R(f, d);
                Mb(f, e);
                R(f, g);
                h ? Mb(f, h) : Mg(f, b);
                k && R(f, k);
                e & 8 || (R(f, l[0]), R(f, l[1]), R(f, l[2]), R(f, l[3]));
                e & 32 && Mg(f, c.id);
                Mb(f, 0);
                return f
            }

            function pr(a, c) {
                var b = c[Wa];
                if (!b || 0 > b || !Df(c) || !c.form || Zh(a, c.form)) return [];
                var d = Ej(a, c.form);
                if (0 > d) return [];
                if (He(c)) {
                    var e = {
                        text: 0,
                        color: 0,
                        Bc: 0,
                        qk: 0,
                        "datetime-local": 0,
                        email: 0,
                        Yf: 0,
                        Lk: 0,
                        search: 0,
                        Rk: 0,
                        time: 0,
                        url: 0,
                        month: 0,
                        Uk: 0,
                        password: 2,
                        Kk: 3,
                        mk: 4,
                        file: 6,
                        image: 7
                    };
                    e = e[c.type]
                } else {
                    e = {fk: 1, gk: 5};
                    var f = Ia(c);
                    e = W(f) ? "" : e[f]
                }
                if ("number" !== typeof e) return [];
                f = -1;
                for (var g = c.form.elements, h = g.length, k = 0, l = 0; k < h; k += 1) if (g[k].name === c.name) {
                    if (g[k] === c) {
                        f = l;
                        break
                    }
                    l += 1
                }
                if (0 > f) return [];
                g = [];
                if (fb(a, g, 7)) return [];
                R(g, b);
                R(g, d);
                R(g, e);
                Lg(g, c.name || "");
                R(g, f);
                return g
            }

            function qr(a, c, b) {
                var d = Ej(a, b);
                if (0 > d) return [];
                var e = b.elements, f = e.length;
                b = [];
                for (var g = 0; g < f; g += 1) if (!Dj(e[g])) {
                    var h = e[g][Wa];
                    h && 0 < h && b.push(h)
                }
                e = [];
                if (fb(a, e, 11)) return [];
                R(e, c);
                R(e, d);
                R(e, b.length);
                for (a = 0; a < b.length; a += 1) R(e, b[a]);
                return e
            }

            function rc(a, c, b) {
                void 0 === b && (b = []);
                for (var d = []; c && !rn(a, c, b); c = Fe(a, c)) d.push(c);
                z(function (e) {
                    sc.Bd += 1;
                    var f = sc.Bd;
                    e[Wa] = f;
                    sc[f] = {};
                    f = nr(a, e);
                    e = pr(a, e);
                    f && e && (Oa(b, f), Oa(b, e))
                }, rr(d));
                return b
            }

            function sr(a) {
                var c = a.sa;
                if (!kd || c && !c.fromElement) return Wh(a)
            }

            function tr(a) {
                var c = a.sa;
                if (c && !c.toElement) return Ef(a)
            }

            function rk(a) {
                var c = zc(a.sa);
                if (c && ie(c)) {
                    var b = Vh(a, c);
                    var d = ub(a.l), e = [];
                    fb(a.l, e, 17) ?
                        a = [] : (R(e, d), R(e, c[Wa]), a = e);
                    return xa(b, a)
                }
            }

            function sk(a) {
                var c = a.l, b = a.sa.target;
                if (b && ie(b)) {
                    c = rc(c, b);
                    var d = ub(a.l), e = [];
                    fb(a.l, e, 18) ? a = [] : (R(e, d), R(e, b[Wa]), a = e);
                    return xa(c, a)
                }
            }

            function tk(a) {
                var c = a.l, b = zc(a.sa);
                if (!b || Bf(c, b) || ld(c, b)) return [];
                if (Df(b)) {
                    var d = H(c).o("isEU"), e = Jc(c, b, d), f = e.cb;
                    d = e.wb;
                    e = e.pb;
                    if (Ie(b)) var g = b.checked; else g = b.value, g = f ? J("", uk(g.split(""))) : g;
                    c = rc(c, b);
                    f = ub(a.l);
                    d = d && !e;
                    e = [];
                    fb(a.l, e, 39) ? a = [] : (R(e, f), R(e, b[Wa]), Mg(e, String(g)), Mb(e, d ? 1 : 0), a = e);
                    return xa(c, a)
                }
            }

            function Te(a) {
                var c = a.l, b = a.sa, d = zc(b);
                if (!d || "SCROLLBAR" === d.nodeName) return [];
                var e = [], f = u(e, Oa);
                d && ie(d) ? f(Vh(a, d)) : f(rc(c, d));
                var g = Vi(c, b);
                a = ub(a.l);
                f = b.type;
                var h = [g.x, g.y];
                g = b.which;
                b = b.button;
                var k;
                var l = Ge(c, d);
                var m = l[0];
                for (l = l[1]; d && (!m || !l);) if (d = Fe(c, d)) l = Ge(c, d), m = l[0], l = l[1];
                d ? (m = d[Wa], !m || 0 > m ? c = [] : (l = (k = {}, k.mousemove = 2, k.click = 32, k.dblclick = 33, k.mousedown = 4, k.mouseup = 30, k.touch = 12, k)[f]) ? (k = [], d = rg(c, d), fb(c, k, l) ? c = [] : (R(k, a), R(k, m), R(k, Math.max(0, h[0] - d.left)), R(k, Math.max(0, h[1] -
                    d.top)), /^mouse(up|down)|click$/.test(f) && (c = g || b, Mb(k, 2 > c ? 1 : c === (g ? 2 : 4) ? 4 : 2)), c = k)) : c = []) : c = [];
                return xa(e, c)
            }

            function ur(a) {
                var c = null, b = a.l, d = b.document;
                if (b.getSelection) {
                    d = void 0;
                    try {
                        d = b.getSelection()
                    } catch (g) {
                        return []
                    }
                    if (Ua(d)) return [];
                    var e = "" + d;
                    c = d.anchorNode
                } else d.selection && d.selection.createRange && (d = d.selection.createRange(), e = d.text, c = d.parentElement());
                if ("string" !== typeof e) return [];
                try {
                    for (; c && 1 !== c.nodeType;) c = c.parentNode
                } catch (g) {
                    return []
                }
                if (!c) return [];
                d = Jc(b, c).cb || ld(b, c, !0);
                c = c.getElementsByTagName("*");
                for (var f = 0; f < c.length && !d;) d = c[f], d = Jc(b, d).cb || ld(b, d, !0), f += 1;
                if (e !== Ng) return Ng = e, d = d ? J("", uk(e.split(""))) : e, e = ub(a.l), 0 === d.length ? d = b = "" : 100 >= d.length ? (b = d, d = "") : 200 >= d.length ? (b = d.substr(0, 100), d = d.substr(100)) : (b = d.substr(0, 97), d = d.substr(d.length - 97)), c = [], fb(a.l, c, 29) ? a = [] : (R(c, e), Lg(c, b), Lg(c, d), a = c), a
            }

            function vr(a) {
                return xa(Te(a), ur(a) || [])
            }

            function vk(a) {
                return (a.shiftKey ? 2 : 0) | (a.ctrlKey ? 4 : 0) | (a.altKey ? 1 : 0) | (a.metaKey ? 8 : 0) | (a.ctrlKey || a.altKey ? 16 : 0)
            }

            function wk(a) {
                var c = [];
                Og || (Og = !0, Ng && c.push.apply(c, mr(a.l, ub(a.l))), Kb(a.l, function () {
                    Og = !1
                }, "fv.c"));
                return c
            }

            function xk(a, c, b, d) {
                c = zc(c);
                if (!c || Gf(a, c)) return [];
                var e = Jc(a, c), f = e.wb, g = e.pb;
                e = e.cb;
                var h = H(a);
                if (!g && (f && h.o("isEU") || ld(a, c))) a = []; else {
                    f = rc(a, c);
                    h = ub(a);
                    g = [];
                    if (fb(a, g, 38)) a = []; else {
                        R(g, h);
                        qk(g, b);
                        Mb(g, d);
                        a = c[Wa];
                        if (!a || 0 > a) a = 0;
                        R(g, a);
                        Mb(g, e ? 1 : 0);
                        a = g
                    }
                    a = xa(f, a)
                }
                return a
            }

            function wr(a) {
                var c = a.l, b = a.sa, d = b.keyCode, e = vk(b), f = [], g = u(f, Oa);
                if ({
                    3: 1,
                    8: 1,
                    9: 1,
                    13: 1,
                    16: 1,
                    17: 1,
                    18: 1,
                    19: 1,
                    20: 1,
                    27: 1,
                    33: 1,
                    34: 1,
                    35: 1,
                    36: 1,
                    37: 1,
                    38: 1,
                    39: 1,
                    40: 1,
                    45: 1,
                    46: 1,
                    91: 1,
                    92: 1,
                    93: 1,
                    106: 1,
                    110: 1,
                    111: 1,
                    144: 1,
                    145: 1
                }[d] || 112 <= d && 123 >= d || 96 <= d && 105 >= d || e & 16) 19 === d && 4 === (e & -17) && (d = 144), g(xk(c, b, d, e | 16)), Pg = !1, Kb(c, function () {
                    Pg = !0
                }, "fv.kd"), !(67 === d && e & 4) || e & 1 || e & 2 || g(wk(a));
                return f
            }

            function xr(a) {
                var c = a.l;
                a = a.sa;
                var b = [];
                Pg && !Qg && 0 !== a.which && (b.push.apply(b, xk(c, a, a.charCode || a.keyCode, vk(a))), Qg = !0, Kb(c, function () {
                    Qg = !1
                }, "fv.kp"));
                return b
            }

            function yk(a) {
                var c = a.l, b = zc(a.sa);
                if (!b || Zh(c, b)) return [];
                var d = [];
                if ("FORM" === b.nodeName) {
                    for (var e = b.elements, f = 0; f < e.length; f += 1) Dj(e[f]) || d.push.apply(d, rc(c, e[f]));
                    d.push.apply(d, qr(c, ub(a.l), b))
                }
                return d
            }

            function yr(a) {
                var c = a.flush;
                a = zc(a.sa);
                "BODY" === Ia(a) && c()
            }

            function Jm(a, c) {
                var b, d = zc(c), e = sa.uc, f = Nd(a);
                if (d && $b("ym-advanced-informer", d)) {
                    var g = f.o("ifc", 0) + 1;
                    f.C("ifc", g);
                    g = d.getAttribute("data-lang");
                    var h = Ga(d.getAttribute("data-cid") || "");
                    if (h || 0 === h) (e = n(a, "Ya." + e + ".informer")) ? e((b = {}, b.i = d, b.id = h, b.lang = g, b)) : f.C("ib", !0), b = c || window.event, b.preventDefault ?
                        b.preventDefault() : b.returnValue = !1
                }
            }

            function qh(a, c, b, d) {
                return function () {
                    var e = Ca(arguments);
                    e = d.apply(void 0, e);
                    return W(e) ? Ha(a, c) : e
                }
            }

            function ph(a, c, b, d) {
                return D(a, "cm." + b, d)
            }

            function El(a, c, b, d, e) {
                return b.length && e ? E(M(function (f, g, h) {
                    return b[h] ? f.concat(E([a, c, d], g)) : f
                }, [], b), t)()(e) : e
            }

            var Sc = {
                    construct: "Metrika2",
                    callbackPostfix: "2",
                    version: "7g4yzra6nxw2gnzhfy8utpb",
                    host: "mc.yandex.ru"
                }, Ig = [], dg = /\./g, hk = Ma(String.prototype.indexOf, "indexOf"), ka = la(function (a, c) {
                    return a === c
                }), gd = la(function (a,
                                      c) {
                    a(c);
                    return c
                }), za = la(fk), Ua = ka(null), W = ka(void 0), Oe = Ma(Array.from, "from"),
                zk = Ma(Function.prototype.bind, "bind"), F = zk ? function () {
                    var a = Ca(arguments);
                    return zk.apply(a[0], xa([a[1]], a.slice(2)))
                } : ar, Oc = la(E), Bi = la(qa), Ak = Ma(Array.prototype.reduce, "reduce"), M = Ak ? function (a, c, b) {
                    return Ak.call(b, a, c)
                } : Ld, kj = t, Qi = t(O, ha), Gg, ek = Pb(window), zr = Ib(ek), Fg = Object.prototype.hasOwnProperty,
                H = x(Nd), X = Ib(n), Pa = X("length"), Kf = Array.prototype.every ? function (a, c) {
                    return Array.prototype.every.call(c, a)
                } : function (a,
                              c) {
                    return M(function (b, d) {
                        return b ? a(d) : !1
                    }, !0, c)
                }, Bk = Ma(Array.prototype.filter, "filter"), Z = Bk ? function (a, c) {
                    return Bk.call(c, a)
                } : dk, Na = u(Boolean, Z), Rg = la(Z), bb = Aa("find", Array.prototype.find) ? function (a, c) {
                    return Array.prototype.find.call(c, a)
                } : Zq, K = Array.prototype.includes ? function (a, c) {
                    return Array.prototype.includes.call(c, a)
                } : Yq, uc = Ib(K), Ck = Ma(Array.prototype.join, "join"), J = Ck ? function (a, c) {
                    return Ck.call(c, a)
                } : Xq, qd = la(J), Dk = x(function (a) {
                    a = n(a, "navigator") || {};
                    var c = n(a, "userAgent") || "";
                    return {
                        Nf: -1 <
                            (n(a, "vendor") || "").indexOf("Apple"), Mg: c
                    }
                }), gb = x(X("navigator.userAgent")), Cg = /Firefox\/([0-9]+)/i, Kd = x(function (a) {
                    var c = n(a, "document.documentElement.style"), b = n(a, "InstallTrigger");
                    a = -1 !== (n(a, "navigator.userAgent") || "").toLowerCase().search(Cg);
                    Cg.lastIndex = 0;
                    return !(!(c && "MozAppearance" in c) || ma(b)) || a
                }), Md, Wq = t(Dg, ka("[object Array]")), Ek = Ma(Array.prototype.map, "map"),
                A = Ek && Vq(window, Array.prototype.map) ? function (a, c) {
                    return c && 0 < c.length ? Ek.call(c, a) : []
                } : ck, z = A, mc = Array.prototype.flatMap ?
                    function (a, c) {
                        return Array.prototype.flatMap.call(c, a)
                    } : Uq, hb = la(A), Jq = Ib(A), Ja = Pe(function (a, c) {
                    return Array.prototype.some.call(c, a)
                }, function (a, c) {
                    for (var b = 0; b < c.length; b += 1) if (b in c && a.call(c, c[b], b)) return !0;
                    return !1
                }, Aa("some", Array.prototype.some)), ye = x(Pb), xc = X("0"), Ar = la(Eg),
                Fk = Ma(Array.prototype.reverse, "reverse"), rr = Fk ? function (a) {
                    return Fk.call(a)
                } : Tq, Gk = Ib(parseInt), Ga = Gk(10), Sg = Gk(2), pa = Object.entries ? function (a) {
                    return a ? Object.entries(a) : []
                } : Zj, da = Object.keys ? Object.keys : ak, Br = t(Zj,
                    u(X("1"), ck)), Cr = Object.values ? Object.values : Br, y = Object.assign || Sq,
                Uh = la(function (a, c) {
                    return y({}, a, c)
                }), bd = x(t(X("String.fromCharCode"), u("fromCharCode", Aa), Tb)),
                Ne = x(t(gb, Ya(/ipad|iphone|ipod/i))), Qf = x(function (a) {
                    return n(a, "navigator.platform") || ""
                }), rd = x(function (a) {
                    a = Dk(a);
                    var c = a.Mg;
                    return a.Nf && !c.match("CriOS")
                }),
                Dr = Ya(/Android.*Version\/[0-9][0-9.]*\sChrome\/[0-9][0-9.]|Android.*Version\/[0-9][0-9.]*\s(?:Mobile\s)?Safari\/[0-9][0-9.]*\sChrome\/[0-9][0-9.]*|; wv\).*Chrome\/[0-9][0-9.]*\sMobile/),
                Er = Ya(/; wv\)/), pd = x(function (a) {
                    a = gb(a);
                    return Er(a) || Dr(a)
                }), Fr = /Chrome\/(\d+)\./, Gr = x(function (a) {
                    return (a = (n(a, "navigator.userAgent") || "").match(Fr)) && a.length ? 76 <= Ga(a[1]) : !1
                }), od = x(function (a) {
                    var c = (gb(a) || "").toLowerCase();
                    a = Qf(a);
                    return !(-1 === c.indexOf("android") || -1 === c.indexOf(c, "mobile") || !/^android|linux armv/i.test(a))
                }), Hr = "other none unknown wifi ethernet bluetooth cellular wimax mixed".split(" "), Ir = x(function (a) {
                    var c = n(a, "navigator.connection.type");
                    if (W(c)) return null;
                    a = ye(a)(c,
                        Hr);
                    return -1 === a ? c : "" + a
                }), sg = x(t(X("document.addEventListener"), Tb)), Hk = x(function (a) {
                    var c = n(a, "navigator") || {};
                    return M(function (b, d) {
                        return b || n(c, d)
                    }, "", ["language", "userLanguage", "browserLanguage", "systemLanguage"])
                }), th = x(function (a) {
                    var c = n(a, "navigator") || {};
                    a = Hk(a);
                    wa(a) || (a = "", c = n(c, "languages.0"), wa(c) && (a = c));
                    return a.toLowerCase().split("-")[0]
                }), eb = x(function (a) {
                    var c = !1;
                    try {
                        c = a.top !== a
                    } catch (b) {
                    }
                    return c
                }), Jr = x(function (a) {
                    var c = !1;
                    try {
                        c = a.top.contentWindow
                    } catch (b) {
                    }
                    return c
                }), Kr =
                    x(function (a) {
                        var c = !1;
                        try {
                            c = a.navigator.javaEnabled()
                        } catch (b) {
                        }
                        return c
                    }), Lr = x(function (a) {
                    var c = "__webdriver_evaluate __selenium_evaluate __webdriver_script_function __webdriver_script_func __webdriver_script_fn __fxdriver_evaluate __driver_unwrapped __webdriver_unwrapped __driver_evaluate __selenium_unwrapped __fxdriver_unwrapped".split(" "),
                        b = n(a, "external");
                    b = -1 !== (n(b, "toString") ? "" + b.toString() : "").indexOf("Sequentum");
                    var d = n(a, "document.documentElement"), e = ["selenium", "webdriver", "driver"];
                    return !!(Ja(u(a, n), ["_selenium", "callSelenium", "_Selenium_IDE_Recorder"]) || Ja(u(n(a, "document"), n), c) || b || d && Ja(F(d.getAttribute, d), e))
                }), Mr = x(function (a) {
                    return !!(Ja(u(a, n), ["_phantom", "__nightmare", "callPhantom"]) || /(PhantomJS)|(HeadlessChrome)/.test(gb(a)) || n(a, "navigator.webdriver") || n(a, "isChrome") && !n(a, "chrome"))
                }), Nr = x(function (a) {
                    return Kf(u(a, n), ["ia_document.shareURL", "ia_document.referrer"])
                }), Qd = x(function (a) {
                    var c = gb(a) || "", b = c.match(/Mac OS X ([0-9]+)_([0-9]+)/);
                    b = b ? [+b[1], +b[2]] : [0,
                        0];
                    c = c.match(/iPhone OS ([1-9]+)_([0-9]+)/);
                    return 14 <= (c ? +c[1] : 0) ? !0 : (Ne(a) || 10 < b[0] || 10 === b[0] && 13 <= b[1]) && rd(a)
                }), Rq = /Edg\/(\d+)\./, ig = x(function (a) {
                    return Qd(a) || pf(a, 68) || qf(a, 79)
                }), Or = Sc.construct, bc = Sc.host, Tg = sg(window), sa = {
                    Zg: 24226447,
                    Tg: 26302566,
                    bh: 51533966,
                    dk: 65446441,
                    Za: "https:",
                    jb: "1060",
                    uc: Or,
                    Yg: Tg ? 512 : 2048,
                    Wg: Tg ? 512 : 2048,
                    Xg: Tg ? 100 : 400,
                    ek: 100,
                    $g: "noindex"
                }, Ue = [], N = x(function (a) {
                    return a.id + ":" + a.ba
                }), gc = {}, Yd = ka("1"), Pr = setTimeout;
            Fa.prototype["catch"] = function (a) {
                return this.then(null,
                    a)
            };
            Fa.prototype.then = function (a, c) {
                var b = new this.constructor(cr);
                jk(this, new er(a, c, b));
                return b
            };
            Fa.prototype["finally"] = function (a) {
                var c = this.constructor;
                return this.then(function (b) {
                    return c.resolve(a()).then(function () {
                        return b
                    })
                }, function (b) {
                    return c.resolve(a()).then(function () {
                        return c.reject(b)
                    })
                })
            };
            Fa.all = function (a) {
                return new Fa(function (c, b) {
                    function d(h, k) {
                        try {
                            if (k && ("object" === typeof k || "function" === typeof k)) {
                                var l = k.then;
                                if ("function" === typeof l) {
                                    l.call(k, function (m) {
                                        d(h, m)
                                    }, b);
                                    return
                                }
                            }
                            e[h] = k;
                            0 === --f && c(e)
                        } catch (m) {
                            b(m)
                        }
                    }

                    if (!a || "undefined" === typeof a.length) return b(new TypeError("Promise.all accepts an array"));
                    var e = Array.prototype.slice.call(a);
                    if (0 === e.length) return c([]);
                    for (var f = e.length, g = 0; g < e.length; g++) d(g, e[g])
                })
            };
            Fa.resolve = function (a) {
                return a && "object" === typeof a && a.constructor === Fa ? a : new Fa(function (c) {
                    c(a)
                })
            };
            Fa.reject = function (a) {
                return new Fa(function (c, b) {
                    b(a)
                })
            };
            Fa.race = function (a) {
                return new Fa(function (c, b) {
                    if (!a || "undefined" === typeof a.length) return b(new TypeError("Promise.race accepts an array"));
                    for (var d = 0, e = a.length; d < e; d++) Fa.resolve(a[d]).then(c, b)
                })
            };
            Fa.bf = "function" === typeof setImmediate && function (a) {
                setImmediate(a)
            } || function (a) {
                Pr(a, 0)
            };
            Fa.dh = function (a) {
                "undefined" !== typeof console && console && console.warn("Possible Unhandled Promise Rejection:", a)
            };
            var I = window.Promise, Qr = Ma(I, "Promise"), Ik = Ma(n(I, "resolve"), "resolve"),
                Jk = Ma(n(I, "reject"), "reject"), Kk = Ma(n(I, "all"), "all");
            if (K(!1, [Qr, Ik, Jk, Kk])) I = Fa; else {
                var Ve = function (a) {
                    return new Promise(a)
                };
                Ve.resolve = F(Ik, I);
                Ve.reject = F(Jk,
                    I);
                Ve.all = F(Kk, I);
                I = Ve
            }
            var Wj = uc([26812653]), eg = x(t(X("id"), Wj), N), Cb = [], U = [], Cc = [], Xd = [], Ug = [], We = [],
                Oq = pb("debuggerEvents", vd),
                Lq = ["http.0.st..rt.", "network error occurred", "send beacon", "Content Security Policy", "DOM Exception 18"],
                Rd, ic = function (a) {
                    return function (c, b) {
                        void 0 === b && (b = !1);
                        if (Rd) var d = new Rd(c); else Aa("Error", a.Error) ? (Rd = a.Error, d = new a.Error(c)) : (Rd = Nq, d = new Rd(c));
                        b && (d.unk = !0);
                        return d
                    }
                }(window), Mq = Ya(/^http./), Kq = Ya(/^err.kn/), Vj = [], Rr = x(function (a) {
                    var c = !1;
                    if (!a.addEventListener) return c;
                    try {
                        var b = Object.defineProperty({}, "passive", {
                            get: function () {
                                c = !0;
                                return 1
                            }
                        });
                        a.addEventListener("test", B, b)
                    } catch (d) {
                    }
                    return c
                }), Sr = la(function (a, c) {
                    return a ? y({capture: !0, passive: !0}, c || {}) : !!c
                }), ia = x(function (a) {
                    a = Rr(a);
                    var c = Sr(a), b = {};
                    return y(b, {
                        D: function (d, e, f, g) {
                            z(function (h) {
                                var k = c(g);
                                Uj(d, h, f, k, !1)
                            }, e);
                            return F(b.kc, b, d, e, f, g)
                        }, kc: function (d, e, f, g) {
                            z(function (h) {
                                var k = c(g);
                                Uj(d, h, f, k, !0)
                            }, e)
                        }
                    })
                }), fa = x(fg), Rj = la(function (a, c) {
                    for (var b = []; !Jd(c);) {
                        var d = Fq(c);
                        a(d, function (e) {
                            return e(c)
                        });
                        b.push(d)
                    }
                    return b
                }), Vg = la(function (a, c) {
                    return za(function (b, d) {
                        return c(b, function (e) {
                            try {
                                d(a(e))
                            } catch (f) {
                                b(f)
                            }
                        })
                    })
                }), Xe = la(function (a, c) {
                    return za(function (b, d) {
                        return c(b, function (e) {
                            try {
                                a(e)(Sa(b, d))
                            } catch (f) {
                                b(f)
                            }
                        })
                    })
                }), yg = [], zg = !1, xg = !1,
                Ub = {id: "id", Ye: "ut", ba: "type", je: "ldc", xb: "nck", Hc: "url", Vh: "referrer"}, Tr = /^\d+$/,
                Tc = {
                    id: function (a) {
                        a = "" + (a || "0");
                        Tr.test(a) || (a = "0");
                        try {
                            var c = Ga(a)
                        } catch (b) {
                            c = 0
                        }
                        return c
                    }, ba: function (a) {
                        return "" + (a || 0 === a ? a : "0")
                    }, xb: Boolean, Ye: Boolean
                };
            Ub.wc = "defer";
            Tc.wc =
                Boolean;
            Ub.ea = "params";
            Tc.ea = function (a) {
                return La(a) || ca(a) ? a : null
            };
            Ub.Xe = "userParams";
            Ub.Hg = "triggerEvent";
            Tc.Hg = Boolean;
            Ub.qg = "sendTitle";
            Tc.qg = function (a) {
                return !!a || W(a)
            };
            Ub.Te = "trackHash";
            Tc.Te = Boolean;
            Ub.Fg = "trackLinks";
            Ub.Fh = "enableAll";
            var gf = M(function (a, c) {
                    var b = c[0];
                    a[b] = {ia: c[1], bb: Tc[b]};
                    return a
                }, {}, pa(Ub)), Lk = la(function (a, c) {
                    var b = c || {};
                    return {
                        l: u(b, O), o: function (d, e) {
                            var f = b[d];
                            return W(f) && !W(e) ? e : f
                        }, C: function (d, e) {
                            b[d] = e;
                            return this
                        }, fc: function (d, e) {
                            return "" === e || ma(e) ? this :
                                this.C(d, e)
                        }, Ca: u(b, a)
                    }
                }), Da = Lk(function (a) {
                    var c = "";
                    a = M(function (b, d) {
                        var e = d[0], f = "" + e + ":" + d[1];
                        "t" === e ? c = f : b.push(f);
                        return b
                    }, [], pa(a));
                    c && a.push(c);
                    return J(":", a)
                }), Wg, Lj = (Wg = {}, Wg.w = [[function (a, c) {
                    return {
                        Z: function (b, d) {
                            var e, f = b.G;
                            f = (e = {}, e["page-url"] = f && f["page-url"] || "", e.charset = "utf-8", e);
                            "0" !== c.ba && (f["cnt-class"] = c.ba);
                            b.H || (b.H = Da());
                            e = b.H;
                            var g = b.Y;
                            f = {
                                ja: {ta: "watch/" + c.id},
                                Y: y(void 0 === g ? {} : g, {qd: !(!e.o("pv") || e.o("ar") || e.o("wh"))}),
                                G: y(b.G || {}, f)
                            };
                            y(b, f);
                            d()
                        }
                    }
                }, 1]], Wg), Mk = u(Lj,
                    Mj), kb = wg("w"), Ij = ["webkitvisibilitychange", "visibilitychange"], ug = Lk(function (a) {
                    a = pa(a);
                    return J("", A(function (c) {
                        var b = c[0];
                        c = c[1];
                        return Ua(c) ? "" : b + "(" + c + ")"
                    }, a))
                }),
                Nk = "A B BIG BODY BUTTON DD DIV DL DT EM FIELDSET FORM H1 H2 H3 H4 H5 H6 HR I IMG INPUT LI OL P PRE SELECT SMALL SPAN STRONG SUB SUP TABLE TBODY TD TEXTAREA TFOOT TH THEAD TR U UL ABBR AREA BLOCKQUOTE CAPTION CENTER CITE CODE CANVAS DFN EMBED FONT INS KBD LEGEND LABEL MAP OBJECT Q S SAMP STRIKE TT ARTICLE AUDIO ASIDE FOOTER HEADER MENU METER NAV PROGRESS SECTION TIME VIDEO NOINDEX NOBR MAIN svg circle clippath ellipse defs foreignobject g glyph glyphref image line lineargradient marker mask path pattern polygon polyline radialgradient rect set text textpath title".split(" "),
                Hp = /^ *(data|javascript):/i,
                Yi = new RegExp(J("", ["\\.(" + J("|", "3gp 7z aac ac3 acs ai avi ape apk asf bmp bz2 cab cdr crc32 css csv cue divx dmg djvu? doc(x|m|b)? emf eps exe flac? flv iso swf gif t?gz jpe?g? js m3u8? m4a mp(3|4|e?g?) m4v md5 mkv mov msi ods og(g|m|v) psd rar rss rtf sea sfv sit sha1 svg tar tif?f torrent ts txt vob wave? wma wmv wmf webm ppt(x|m|b)? xls(x|m|b)? pdf phps png xpi g?zip".split(" ")) + ")$"]), "i"),
                Qa, kk = (Qa = {}, Qa.hit = "h", Qa.params = "p", Qa.reachGoal = "g", Qa.userParams = "up",
                    Qa.trackHash = "th", Qa.accurateTrackBounce = "atb", Qa.notBounce = "nb", Qa.addFileExtension = "fe", Qa.extLink = "el", Qa.file = "fc", Qa.trackLinks = "tl", Qa.destruct = "d", Qa.setUserID = "ui", Qa.getClientID = "ci", Qa.clickmap = "cm", Qa.enableAll = "ea", Qa),
                Ur = x(function () {
                    var a = 0;
                    return function () {
                        return a += 1
                    }
                }), Vr = t(N, Ur, ha), Jb = la(function (a, c) {
                    return H(c).o(a, null)
                }), Gd = {
                    mc: function (a) {
                        a = Nd(a).o("mt", {});
                        a = pa(a);
                        return a.length ? M(function (c, b, d) {
                            return "" + c + (d ? "-" : "") + b[0] + "-" + b[1]
                        }, "", a) : null
                    }, clc: function (a) {
                        var c = H(a).o("cls",
                            {tc: 0, x: 0, y: 0}), b = c.tc, d = c.x;
                        c = c.y;
                        return b ? b + "-" + a.Math.floor(d / b) + "-" + a.Math.floor(c / b) : b + "-" + d + "-" + c
                    }, rqnt: function (a, c, b) {
                        a = b.G;
                        return !a || a.nohit ? null : Vr(c)
                    }
                };
            Gd.oc = Jb("oc");
            Gd.oca = Jb("oca");
            var Fb = E([1, null], Pe), hd = E([1, 0], Pe), xq = x(function (a) {
                    Gj(a, "_ymBRC", "1");
                    var c = "1" !== Fj(a, "_ymBRC");
                    c || Hj(a, "_ymBRC");
                    return c
                }), Ra = x(sf), Uc = x(sf, function (a, c, b) {
                    return "" + c + b
                }), Wr = x(function (a) {
                    a = n(a, "document") || {};
                    return ("" + (a.characterSet || a.charset || "")).toLowerCase()
                }), ab = x(t(X("document"), u("createElement",
                    ec))), $h = x(function (a) {
                    var c = n(a, "Element.prototype");
                    return c ? (a = bb(function (b) {
                        return Aa(b, c[b])
                    }, ["matches", "webkitMatchesSelector", "mozMatchesSelector", "msMatchesSelector", "oMatchesSelector"])) ? c[a] : null : null
                }), Xr = ka("INPUT"), He = t(Ia, Xr), Yr = ka("TEXTAREA"), uq = t(Ia, Yr), Zr = ka("SELECT"),
                vq = t(Ia, Zr), Ie = t(X("type"), Ya(/^(checkbox|radio)$/)), Df = t(Ia, Ya(/^INPUT|SELECT|TEXTAREA$/)),
                ie = t(Ia, Ya(/^INPUT|SELECT|TEXTAREA|BUTTON$/)),
                Hf = "INPUT CHECKBOX RADIO TEXTAREA SELECT PROGRESS".split(" "), tq = ["submit",
                    "image", "hidden"], mj = /^\s+|\s+$/g, Bj = Ma(String.prototype.trim, "trim"), Ok = la(function (a, c) {
                    return c.replace(a, "")
                }), Fi = Ok(/\s/g), Ob = Ok(/\D/g), Uf = x(function () {
                    for (var a = 59, c = {}, b = 0; b < Nk.length; b += 1) c[Nk[b]] = String.fromCharCode(a), a += 1;
                    return c
                }), zj = x(function (a) {
                    return {wk: a, qb: null, zb: []}
                }), xj = {}, pg = {};
            xj.p = 500;
            var wj = {i: "id", n: "name", h: "href", ty: "type"};
            pg.h = !0;
            pg.c = !0;
            var Pc = {};
            Pc.p = qg;
            Pc.c = function (a, c, b) {
                (a = ob(n(c, "textContent"))) && b && (b = b(c), b.length && Ja(t(X("textContent"), ob, ka(a)), b) && (a = ""));
                He(c) && (a = ob(c.getAttribute && c.getAttribute("value") || a));
                return a
            };
            var Vc, og = "button," + A(function (a) {
                    return 'input[type="' + a + '"]'
                }, ["button", "submit", "reset", "file"]).join(",") + ",a", Wf = u(og, xb),
                qq = (Vc = {}, Vc.A = "h", Vc.BUTTON = "i", Vc.DIV = "i", Vc.INPUT = "ty", Vc),
                pq = "hash host hostname href pathname port protocol search".split(" "),
                ng = "ru ua by kz az kg lv md tj tm uz ee fr lt com co.il com.ge com.am com.tr com.ua com.ru".split(" "),
                vj = /(?:^|\.)(?:(ya\.ru)|(?:yandex)\.(\w+|com?\.\w+))$/, we = x(function (a) {
                    return (a ?
                        a.replace(/^www\./, "") : "").toLowerCase()
                }), $r = x(function (a) {
                    a = S(a).hostname;
                    var c = !1;
                    a && (c = -1 !== a.search(vj));
                    return c
                }), Pk = t(S, X("protocol"), ka("https:")), Qk = /\/$/, as = x(t(fa, za(function (a) {
                    return -(new a.l.Date).getTimezoneOffset()
                }))), bs = t(fa, za(function (a) {
                    a = new a.l.Date;
                    return J("", A(Gq, [a.getFullYear(), a.getMonth() + 1, a.getDate(), a.getHours(), a.getMinutes(), a.getSeconds()]))
                })), cs = t(fa, za(jg)), Rk = x(t(fa, za(function (a) {
                    return a.Ba[0]
                }))), ds = x(Ac), nq = x(function (a) {
                    return Gr(a) && Pk(a) ? "SameSite=None;Secure;" :
                        ""
                }), mg = ["metrika_enabled"], lg = [], tj = pb("gsc", rj), oq = /:\d+$/, fr = x(function (a) {
                    var c = (S(a).host || "").split(".");
                    return 1 === c.length ? c[0] : M(function (b, d, e) {
                        e += 1;
                        2 <= e && !b && (e = J(".", c.slice(-e)), ai(a, e) && (b = e));
                        return b
                    }, "", c)
                }), ac = x(Dc), qj = pb("r", function (a, c) {
                    var b = pj(a, c), d = b[0];
                    return !b[1] && d
                }), Hd = x(function () {
                    return {Ja: {}, pending: {}, children: {}}
                }), Xg = X("postMessage"), es = C("s.f", function (a, c, b, d, e) {
                    c = c(d);
                    var f = Hd(a), g = J(":", [c.aa.Bc, c.aa.key]);
                    if (Xg(b)) {
                        f.pending[g] = e;
                        try {
                            b.postMessage(c.Ag, "*")
                        } catch (h) {
                            delete f.pending[g];
                            return
                        }
                        V(a, function () {
                            delete f.pending[g]
                        }, 5E3, "if.s")
                    }
                }), fs = C("s.fh", function (a, c, b, d, e, f) {
                    var g = null, h = null, k = Hd(a), l = null;
                    try {
                        g = tb(a, f.data), h = g.__yminfo, l = g.data
                    } catch (m) {
                        return
                    }
                    if (!ma(h) && h.substring && "__yminfo" === h.substring(0, 8) && !ma(l) && (g = h.split(":"), 4 === g.length)) if (h = c.id, c = g[1], a = g[2], g = g[3], !ca(l) && l.type && "0" === g && l.counterId) {
                        if (!l.toCounter || l.toCounter == h) {
                            k = null;
                            try {
                                k = f.source
                            } catch (m) {
                            }
                            !Ua(k) && Xg(k) && (f = d.O(l.type, [f, l]), e = A(t(O, Uh(e)), f.concat([{}])), l = b([c, a, l.counterId], e),
                                k.postMessage(l.Ag, "*"))
                        }
                    } else g === "" + h && ca(l) && Z(function (m) {
                        return !(!m.hid || !m.counterId)
                    }, l).length === l.length && (b = k.pending[J(":", [c, a])]) && b.apply(null, [f].concat(l))
                }), cd = x(function (a, c) {
                    var b, d = ec("getElementsByTagName", n(a, "document")), e = Hd(a), f = Xg(a), g = id(a), h = ia(a);
                    if (!d || !f) return null;
                    d = d.call(a.document, "iframe");
                    f = (b = {}, b.counterId = c.id, b.hid = "" + Lb(a), b);
                    ig(a) && (f.duid = Fc(a, c));
                    jq(a, g);
                    kq(a);
                    b = lq(a, f);
                    var k = E([a, u([], b)], es);
                    z(function (l) {
                        var m = null;
                        try {
                            m = l.contentWindow
                        } catch (p) {
                        }
                        m &&
                        k(m, {type: "initToChild"}, function (p, q) {
                            g.O("initToParent", [p, q])
                        })
                    }, d);
                    eb(a) && k(a.parent, {type: "initToParent"}, function (l, m) {
                        g.O("parentConnect", [l, m])
                    });
                    h.D(a, ["message"], E([a, c, b, g, f], fs));
                    return {ca: g, Ja: e.Ja, children: e.children, Je: k}
                }, t(yb, N)), dd = x(function (a, c) {
                    if (!ig(a) || !eb(a)) return Fc(a, c);
                    var b = cd(a, c);
                    return b && b.Ja[c.id] ? b.Ja[c.id].info.duid || Fc(a, c) : Fc(a, c)
                }, function (a, c) {
                    return "{" + c.je + c.xb
                }), gs = x(function (a) {
                    a = H(a);
                    var c = a.o("counterNum", 0) + 1;
                    a.C("counterNum", c);
                    return c
                }, t(yb, N)), ja,
                Ee = (ja = {}, ja.vf = u(Sc.version, O), ja.nt = Ir, ja.fp = function (a, c, b) {
                    if (b.G && b.G.nohit) return null;
                    c = N(c);
                    b = ds(a);
                    if (b[c]) return null;
                    a:{
                        var d = Rk(a), e = n(a, "performance.getEntriesByType");
                        if (T(e)) {
                            if (a = Z(t(O, X("name"), ka("first-contentful-paint")), e.call(a.performance, "paint")), a.length) {
                                a = a[0].startTime;
                                break a
                            }
                        } else {
                            e = n(a, "chrome.loadTimes");
                            if (T(e) && (e = e.call(a.chrome), e = n(e, "firstPaintTime"), d && e)) {
                                a = 1E3 * e - d;
                                break a
                            }
                            if (a = n(a, "performance.timing.msFirstPaint")) {
                                a -= d;
                                break a
                            }
                        }
                        a = void 0
                    }
                    return a ? (b[c] = a,
                        Math.round(a)) : null
                }, ja.fu = function (a, c, b) {
                    var d = b.G;
                    if (!d) return null;
                    c = (n(a, "document.referrer") || "").replace(Qk, "");
                    b = (d["page-ref"] || "").replace(Qk, "");
                    d = d["page-url"];
                    a = S(a).href !== d;
                    c = c !== b;
                    b = 0;
                    a && c ? b = 3 : c ? b = 1 : a && (b = 2);
                    return b
                }, ja.en = Wr, ja.la = Hk, ja.ut = function (a, c, b) {
                    var d = b.V;
                    b = b.G;
                    d = d && d.Uc;
                    b && ($r(a) || c.Ye || d) && (b.ut = sa.$g);
                    return null
                }, ja.v = u(sa.jb, O), ja.cn = gs, ja.dp = function (a) {
                    var c = H(a), b = c.o("bt", {});
                    if (W(c.o("bt"))) {
                        var d = n(a, "navigator.getBattery");
                        try {
                            b.p = d && d.call(a.navigator)
                        } catch (e) {
                        }
                        c.C("bt",
                            b);
                        b.p && b.p.then && b.p.then(D(a, "bi:dp.p", function (e) {
                            b.Wj = n(e, "charging") && 0 === n(e, "chargingTime")
                        }))
                    }
                    return hd(b.Wj)
                }, ja.ls = x(function (a, c) {
                    var b = Uc(a, c.id), d = fa(a), e = b.o("lsid");
                    return +e ? e : (d = Va(a, 0, d(aa)), b.C("lsid", d), d)
                }, yb), ja.hid = Lb, ja.phid = function (a, c) {
                    if (!eb(a)) return null;
                    var b = cd(a, c);
                    if (!b) return null;
                    var d = da(b.Ja);
                    return d.length ? b.Ja[d[0]].info.hid : null
                }, ja.z = as, ja.i = bs, ja.et = cs, ja.c = t(X("navigator.cookieEnabled"), Fb), ja.rn = t(O, Va), ja.rqn = function (a, c, b) {
                    b = b.G;
                    if (!b || b.nohit) return null;
                    c = N(c);
                    a = Uc(a, c);
                    c = (a.o("reqNum", 0) || 0) + 1;
                    a.C("reqNum", c);
                    if (a.o("reqNum") === c) return c;
                    a.Rb("reqNum");
                    return null
                }, ja.u = dd, ja.w = function (a) {
                    a = Qc(a);
                    return a[0] + "x" + a[1]
                }, ja.s = function (a) {
                    var c = n(a, "screen");
                    if (c) {
                        a = n(c, "width");
                        var b = n(c, "height");
                        c = n(c, "colorDepth") || n(c, "pixelDepth");
                        return J("x", [a, b, c])
                    }
                    return null
                }, ja.sk = X("devicePixelRatio"), ja.ifr = t(eb, Fb), ja.j = t(Kr, Fb), ja.sti = function (a) {
                    return eb(a) ? Jr(a) ? "1" : null : null
                }, ja), iq = x(function () {
                    return Oa(da(Ee), da(Gd))
                }), hq = x(Ac, N), nj = x(function () {
                    return {
                        wf: null,
                        wa: []
                    }
                }, N), eq = /^[a-z][\w.+-]+:/i, Yg, Rb = [[Ke, 1], [De, 2], [Hb(), 3], [oj, 4]], Ce = [], zb = u(Rb, Nj),
                Qb = (Yg = {}, Yg.h = Rb, Yg), ba = u(Qb, Mj);
            zb(function (a) {
                return {
                    Z: function (c, b) {
                        var d = c.G;
                        if (!c.H || !d) return b();
                        var e = d["page-ref"], f = d["page-url"];
                        e && f !== e ? d["page-ref"] = lj(a, e) : delete d["page-ref"];
                        d["page-url"] = lj(a, f).slice(0, sa.Yg);
                        return b()
                    }
                }
            }, -100);
            var aq = /[^a-z0-9.:-]/, Zg, $g = {}, Sk = Na([gg && [gg, 0], 0, [wb, 2], Fd && [Fd, 3], [Nc, 4]]),
                Wc = Na([gg, 0, wb, Fd, Nc]), Tk = [wb];
            Tk.push(Fd);
            var Uk = Na(Tk), Xc = Na([Nc]);
            Na([0, wb]);
            var hs =
                    Na([0, Nc]), Ye = Na([0, wb, Fd, Nc]), va = (Zg = {}, Zg.h = Uk, Zg), ed = x(function (a, c) {
                    var b = $g["*"] ? $g["*"] : c && $g[c];
                    b || (b = c ? va[c] || [] : Wc);
                    b = M(function (d, e) {
                        var f = e(a);
                        if (f) {
                            var g = bb(t(xc, ka(e)), Sk);
                            g && d.push([g[1], f])
                        }
                        return d
                    }, [], b);
                    b.length || ff();
                    return b
                }, yb), ah, is = F(I.reject, I, Ta()), Ea = (ah = {}, ah.h = kb, ah), Ba = C("g.sen", function (a, c, b) {
                    var d = ed(a, c);
                    b = b ? dq(a, c, b) : [];
                    var e = Ea[c], f = e ? e(a, d, b) : kb(a, d, b);
                    return function () {
                        var g = Ca(arguments), h = g[0];
                        g = g.slice(1);
                        var k = h.Y;
                        h = y(h, {Y: y(void 0 === k ? {} : k, {Fa: [c]})});
                        return f.apply(void 0,
                            xa([h], g))
                    }
                }, is), Up = la(function (a, c) {
                    if (!c[a]) {
                        var b, d = new I(function (e) {
                            b = e
                        });
                        c[a] = {hg: b, gb: d, ig: !1}
                    }
                    return c[a].gb
                }), ij = x(t(Ac, za)), Sd = x(function (a, c) {
                    var b = n(a, "console"), d = n(b, "log");
                    d = Qe("log", d) ? F(d, b) : B;
                    var e = n(b, "warn");
                    e = Qe("warn", e) ? F(e, b) : d;
                    var f = n(b, "error");
                    b = Qe("error", f) ? F(f, b) : d;
                    return {log: Mc(a, "log", c, d), error: Mc(a, "error", c, b), warn: Mc(a, "warn", c, e)}
                }), js = C("dc.init", function (a, c) {
                    function b(k) {
                        for (var l = [], m = 1; m < arguments.length; m++) l[m - 1] = arguments[m];
                        H(a).o("dce:" + c, !1) && e[k].apply(e,
                            l);
                        H(a).o("dclq:" + c).push([k, l])
                    }

                    var d = S(a), e = Sd(a, c);
                    H(a).Va("dclq:" + c, []);
                    var f = ac(a), g = Jf(a), h = g.Ii;
                    g = g.yi;
                    h && !g && f.C("debug", "1", void 0, d.host);
                    return h || g ? {log: u("log", b), warn: u("warn", b), error: u("error", b)} : Tp(a, c)
                }), Dd = x(js, yb), ks = C("p.dc", function (a, c) {
                    var b = N(c);
                    hj(a, "");
                    hj(a, b)
                }), Hl = D(window, "h.p", function (a, c) {
                    var b, d, e = Ba(a, "h", c), f = c.Hc || "" + S(a).href, g = c.Vh || a.document.referrer,
                        h = {H: Da((b = {}, b.pv = 1, b)), G: (d = {}, d["page-url"] = f, d["page-ref"] = g, d), V: {}};
                    h.V.ea = c.ea;
                    h.V.Xe = c.Xe;
                    c.wc && h.G &&
                    (h.G.nohit = "1");
                    return e(h, c).then(function (k) {
                        k && (c.wc || Gb(a, c, "PageView. Counter " + c.id + ". URL: " + f + ". Referrer: " + g, c.ea)(), Kb(a, E([a, c, k], Vp)))
                    })["catch"](D(a, "h.g.s"))
                }), bh = ["yandex_metrika_callback" + Sc.callbackPostfix, "yandex_metrika_callbacks" + Sc.callbackPostfix],
                ls = C("cb.i", function (a) {
                    var c = bh[0], b = bh[1];
                    if (T(a[c])) a[c]();
                    "object" === typeof a[b] && z(function (d, e) {
                        a[b][e] = null;
                        ag(a, d)
                    }, a[b]);
                    z(function (d) {
                        try {
                            delete a[d]
                        } catch (e) {
                            a[d] = void 0
                        }
                    }, bh)
                }), Vk = x(function (a) {
                    return n(a, "crypto.subtle.digest") &&
                        n(a, "TextEncoder") && n(a, "FileReader") && n(a, "Blob")
                }), ms = C("fpm", function (a, c) {
                    if (!Pk(a)) return B;
                    var b = N(c);
                    if (!Vk(a)) return Db(a, b, "Not supported"), B;
                    var d = Ha(a, c);
                    return d ? function (e) {
                        return (new I(function (f, g) {
                            return La(e) ? da(e).length ? f(ej(a, e).then(function (h) {
                                var k, l;
                                h && h.length && d.params((k = {}, k.__ym = (l = {}, l.fpp = h, l), k))
                            }, B)) : g(Ta("fpm.l")) : g(Ta("fpm.o"))
                        }))["catch"](D(a, "fpm.en"))
                    } : B
                }), Ze = la(function (a, c) {
                    var b = {};
                    cg(a)(function (d) {
                        b = d[c] || {}
                    });
                    return b
                }), ns = C("c.c.cc", function (a) {
                    var c = H(a),
                        b = t(Ze(a), function (d) {
                            var e, f = (e = {}, e.clickmap = !!d.clickmap, e), g = !!a.ya_cid;
                            g && c.C("oc", 1);
                            e = {};
                            try {
                                a.Object.defineProperty(e, "oldCode", {
                                    get: function () {
                                        c.C("oca", 1);
                                        return g
                                    }
                                })
                            } catch (h) {
                                f.oldCode = g, c.C("oca", 0)
                            }
                            return y(e, d, f)
                        });
                    return D(a, "g.c.cc", t(F(c.o, c, "counters", {}), da, hb(b)))
                }), os = C("gt.c.rs", function (a, c) {
                    var b, d = N(c), e = c.id, f = c.ba, g = c.vh, h = c.Te, k = E([a, d], Pp);
                    bg(a, d, (b = {}, b.id = e, b.type = +f, b.clickmap = g, b.trackHash = !!h, b));
                    return k
                }), cj = x(vd), Cd = x(Ac, N), ps = C("pa.int", function (a, c) {
                    var b;
                    return b =
                        {}, b.params = function () {
                        var d, e, f = Ca(arguments), g = Op(f);
                        if (!g) return null;
                        f = g.Ah;
                        var h = g.ea;
                        g = g.pc;
                        if (!La(h) && !ca(h)) return null;
                        var k = Ba(a, "1", c), l = Cd(c).url, m = !eg(c), p = "arams. Counter " + c.id, q = "P" + p, r = h,
                            v = "";
                        (v = n(h, "__ym.user_id")) && (q = "Set user id " + v);
                        K("__ymu", da(h)) && (q = "User p" + p);
                        r.__ym && (r = y({}, h), r.__ym = M(function (w, G) {
                            var Y = n(h, "__ym." + G);
                            Y && (w[G] = Y);
                            return w
                        }, {}, Ue), da(r.__ym).length || delete r.__ym, m = !!da(r).length);
                        r = v ? void 0 : JSON.stringify(r);
                        p = Gb(a, c, q, r);
                        k = k({
                            V: {ea: h}, H: Da((d = {}, d.pa =
                                1, d.ar = 1, d)), G: (e = {}, e["page-url"] = l || S(a).href, e)
                        }, c).then(m ? p : B);
                        return Lc(a, "p.s", k, g, f)
                    }, b
                }), de = x($i, t(yb, N)), qs = C("y.p", function (a, c) {
                    var b = $i(a, c);
                    if (b) {
                        var d = Zd(a), e = E([a, b, c], Kp);
                        wh(a, d, function (f) {
                            f.D(["params"], e)
                        });
                        b.ca.D(["params"], t(X("1"), e))
                    }
                }), gr = x(function (a) {
                    if (a = ab(a)) return a("a")
                }), Wk = {zk: Ya(/[/&=?#]/)}, te = C("go.in", function (a, c, b, d) {
                    var e;
                    void 0 === b && (b = "goal");
                    return e = {}, e.reachGoal = function (f, g, h, k) {
                        var l, m;
                        if (!f || Wk[b] && Wk[b](f)) return null;
                        var p = g, q = h || B;
                        T(g) && (q = g, p = void 0,
                            k = h);
                        var r = Gb(a, c, "Reach goal. Counter: " + c.id + ". Goal id: " + f, p), v = "goal" === b;
                        g = Ba(a, "g", c);
                        var w = Jp(a, c, f, b);
                        h = w[0];
                        w = w[1];
                        p = g({
                            V: {ea: p},
                            H: Da((l = {}, l.ar = 1, l)),
                            G: (m = {}, m["page-url"] = h, m["page-ref"] = w, m)
                        }, c).then(function () {
                            v && r();
                            ib(a, {da: N(c), name: "event", data: {Eb: b, name: f}});
                            d && d()
                        });
                        return Lc(a, "g.s", p, q, k)
                    }, e
                }), rs = C("guid.int", function (a, c) {
                    var b;
                    return b = {}, b.getClientID = function (d) {
                        var e = Fc(a, c);
                        d && ag(a, d, null, e);
                        return e
                    }, b
                }), mk, ss = C("th.e", function (a, c) {
                    function b() {
                        g || (k = Ad(a, "onhashchange") ?
                            ia(a).D(a, ["hashchange"], h) : hr(a, h))
                    }

                    var d, e = Ba(a, "t", c), f = ze(a, N(c)), g = !1, h = D(a, "h.h.ch", F(ir, null, a, c, e)), k = B;
                    c.Te && (b(), g = !0);
                    e = D(a, "tr.hs.h", function (l) {
                        var m;
                        l ? b() : k();
                        g = !!l;
                        f((m = {}, m.trackHash = g, m))
                    });
                    return d = {}, d.trackHash = e, d.u = k, d
                }), ts = la(function (a, c) {
                    wa(c) ? a.push(c) : z(t(O, qa("push", a)), c)
                }), us = C("cl.p", function (a, c) {
                    function b(p, q, r, v) {
                        void 0 === v && (v = {});
                        r ? ve(a, c, {url: r, ub: !0, Pc: p, Uc: q, sender: e, Ng: v}) : g.warn("Empty link")
                    }

                    var d, e = Ba(a, "2", c), f = [], g = Sd(a, N(c)), h = N(c), k = D(a, "s.s.tr", u(ze(a,
                        h), Ip));
                    h = {
                        l: a,
                        kb: c,
                        Nh: f,
                        sender: e,
                        globalStorage: H(a),
                        zh: Uc(a, c.id),
                        Ak: Lb(a),
                        Oj: u(u(h, Ze(a)), t(ha, X("trackLinks")))
                    };
                    h = D(a, "cl.p.c", u(h, Fp));
                    h = ia(a).D(a, ["click"], h);
                    c.Fg && k(c.Fg);
                    var l = D(a, "file.clc", E([!0, !1], b)), m = D(a, "e.l.l.clc", E([!1, !0], b));
                    f = D(a, "add.f.e.clc", ts(f));
                    return d = {}, d.file = l, d.extLink = m, d.addFileExtension = f, d.trackLinks = k, d.u = h, d
                }), Bd = pb("retryReqs", function (a) {
                    var c = Ra(a), b = c.o("retryReqs", {}), d = fa(a)(aa);
                    z(function (e) {
                            var f = e[0];
                            e = e[1];
                            (!e || !e.time || e.time + 864E5 < d) && delete b[f]
                        },
                        pa(b));
                    c.C("retryReqs", b);
                    return b
                }, !0), Xk = Ib(t(he, ka(0))), vs = [Xk("watch"), Xk("clmap")], ws = C("g.r", function (a) {
                    var c = fa(a), b = Bd(a), d = c(aa), e = Lb(a);
                    return M(function (f, g) {
                        var h = g[0], k = g[1];
                        k && Ja(za(k.resource), vs) && !k.d && k.ghid && k.ghid !== e && k.time && 500 < d - k.time && k.time + 864E5 > d && 2 >= k.browserInfo.rqnl && (k.d = 1, h = {
                            protocol: k.protocol,
                            host: k.host,
                            ta: k.resource,
                            nj: k.postParams,
                            ea: k.params,
                            lh: k.browserInfo,
                            yk: k.ghid,
                            time: k.time,
                            cc: Ga(h),
                            yh: k.counterId,
                            ba: k.counterType
                        }, k.telemetry && (h.La = k.telemetry), f.push(h));
                        return f
                    }, [], pa(b))
                }), xs = C("nb.p", function (a, c) {
                    function b(G) {
                        l() || (G = "number" === typeof G ? G : 15E3, w = jr(a, d(!1), G), m())
                    }

                    function d(G) {
                        return function (Y) {
                            var Q, oa, ta;
                            void 0 === Y && (Y = (Q = {}, Q.ctx = {}, Q.callback = B, Q));
                            if (G || !r && !k.$d) {
                                r = !0;
                                m();
                                w && w();
                                var vb = p(aa);
                                Q = (Ga(k.o("lastHit")) || 0) < vb - 18E5;
                                var ud = .1 > Math.random();
                                k.C("lastHit", vb);
                                vb = Da((oa = {}, oa.nb = 1, oa.cl = v, oa.ar = 1, oa));
                                oa = Cd(c);
                                oa = {G: (ta = {}, ta["page-url"] = oa.url || S(a).href, ta), H: vb, V: {force: G}};
                                ta = Sd(a, N(c)).warn;
                                !Y.callback && Y.ctx && ta('"callback" argument missing');
                                (ta = G || Q || ud) || (ta = a.location.href, Q = a.document.referrer, ta = !(ta && Q ? Zi(ta) === Zi(Q) : !ta && !Q));
                                if (ta) return ta = g(oa, c), Lc(a, "l.o.l", ta, Y.callback, Y.ctx)
                            }
                            return null
                        }
                    }

                    var e, f, g = Ba(a, "n", c), h = N(c), k = Uc(a, c.id),
                        l = u(u(h, Ze(a)), t(ha, X("accurateTrackBounce"))),
                        m = u((e = {}, e.accurateTrackBounce = !0, e), ze(a, h)), p = fa(a), q = p(aa), r = !1, v = 0, w;
                    ra(c, function (G) {
                        v = G.Ph - q
                    });
                    c.df && b(c.df);
                    e = (f = {}, f.notBounce = d(!0), f.u = w, f);
                    e.accurateTrackBounce = b;
                    return e
                }), Cp = la($b)("(ym-disable-clickmap|ym-clickmap-ignore)"), ys = C("clm.p",
                    function (a, c) {
                        if (bd(a)) return B;
                        var b = Ba(a, "m", c), d = N(c), e = fa(a), f = e(aa), g = u(u(d, Ze(a)), t(ha, X("clickmap"))), h,
                            k = null;
                        d = D(a, "clm.p.c", function (l) {
                            var m = g();
                            if (m) {
                                var p = H(a), q = p.o("cls", {tc: 0, x: 0, y: 0});
                                p.C("cls", {tc: q.tc + 1, x: q.x + l.clientX, y: q.y + l.clientY});
                                p = "object" === typeof m ? m : {};
                                q = p.filter;
                                m = p.isTrackHash || !1;
                                var r = A(function (w) {
                                    return ("" + w).toUpperCase()
                                }, p.ignoreTags || []);
                                W(h) && (h = p.quota || null);
                                var v = !!p.quota;
                                l = {element: Dp(a, l), position: Vi(a, l), button: Ep(l), time: e(aa)};
                                p = S(a).href;
                                if (Bp(a,
                                    l, k, r, q)) {
                                    if (v) {
                                        if (!h) return;
                                        --h
                                    }
                                    r = Ge(a, l.element);
                                    q = r[0];
                                    r = r[1];
                                    v = rg(a, l.element);
                                    q = ["rn", Va(a), "x", Math.floor(65535 * (l.position.x - v.left) / (q || 1)), "y", Math.floor(65535 * (l.position.y - v.top) / (r || 1)), "t", Math.floor((l.time - f) / 100), "p", qg(a, l.element), "X", l.position.x, "Y", l.position.y];
                                    q = J(":", q);
                                    m && (q += ":wh:1");
                                    Ap(a, p, q, b, c);
                                    k = l
                                }
                            }
                        });
                        return ia(a).D(n(a, "document"), ["click"], d)
                    }), zs = C("trigger.in", function (a, c) {
                    c.Hg && Kb(a, E([a, "yacounter" + c.id + "inited"], wq), "t.i")
                }), As = C("c.m.p", function (a, c) {
                    var b, d =
                        N(c);
                    return b = {}, b.clickmap = u(ze(a, d), zp), b
                }), ui = u("form", dc), jp = u("form", xb), yp = x(function (a, c) {
                    return ra(c, X("settings.form_goals"))
                }, yb), Bs = u(!0, Si), Cs = C("s.f.i", function (a, c) {
                    var b = [];
                    ia(a).D(a, ["click"], D(a, "s.f.c", E([a, c, b], xp)));
                    ia(a).D(a, ["submit"], D(a, "s.f.e", t(X("target"), E([a, c, b], Bs))));
                    Ti(a, c, "Form goal. Counter " + c.id + ". Init.")
                }), Ds = C("s.f.i", function (a, c) {
                    return ra(c, function (b) {
                        if (n(b, "settings.button_goals") || -1 !== S(a).href.indexOf("yagoalsbuttons=1")) ia(a).D(a, ["click"], D(a, "c.t.c",
                            t(X("target"), E([a, c], ef(a, c, "", wp))))), Gb(a, c, "Button goal. Counter " + c.id + ". Init.")()
                    })
                }), Vb, Td, ch, Bc,
                Yf = (Vb = {}, Vb.transaction_id = "id", Vb.item_brand = "brand", Vb.index = "position", Vb.item_variant = "variant", Vb.value = "revenue", Vb.item_category = "category", Vb.item_list_name = "list", Vb),
                vc = (Td = {}, Td.item_id = "id", Td.item_name = "name", Td.promotion_name = "coupon", Td),
                vp = (ch = {}, ch.promotion_name = "name", ch),
                Ri = "currencyCode add delete remove purchase checkout detail".split(" "),
                yd = (Bc = {}, Bc.view_item = {
                    event: "detail",
                    za: vc, Ma: "products"
                }, Bc.add_to_cart = {event: "add", za: vc, Ma: "products"}, Bc.remove_from_cart = {
                    event: "remove",
                    za: vc,
                    Ma: "products"
                }, Bc.begin_checkout = {event: "checkout", za: vc, Ma: "products"}, Bc.purchase = {
                    event: "purchase",
                    za: vc,
                    Ma: "products"
                }, Bc), Pi = C("dl.w", function (a, c, b) {
                    function d() {
                        var g = n(a, c);
                        (e = ca(g) && xe(a, g, b)) || (f = V(a, d, 1E3, "ec.dl"))
                    }

                    var e, f = 0;
                    d();
                    return function () {
                        return na(a, f)
                    }
                }), Es = C("p.e", function (a, c) {
                    var b = Ha(a, c);
                    if (b) {
                        var d = H(a);
                        b = b.params;
                        var e = D(a, "h.ee", E([a, N(c), b], sp));
                        return c.Hd ?
                            (d.C("ecs", 0), Oi(a, c.Hd, e)) : ra(c, function (f) {
                                if (f = n(f, "settings.ecommerce")) return d.C("ecs", 1), Oi(a, f, e)
                            })
                    }
                }), Li = x(function (a) {
                    return J("[^\\d<>]*", a.split(""))
                }), zm = x(function (a) {
                    return new RegExp(Li(a), "g")
                }), pp = /\S/,
                Ei = u(["style", "display:inline;margin:0;padding:0;font-size:inherit;color:inherit;line-height:inherit"], Gc),
                Yk = x(function (a) {
                    return bd(a) || !Id(a)
                }), Fs = C("phc.h", function (a, c) {
                    return Yj(a) || Yk(a) ? null : ra(c, function (b) {
                        if (!n(b, "settings.phchange")) {
                            var d = Dc(a, "").o("yaHidePhones");
                            d =
                                d ? tb(a, d) : "";
                            (b = n(b, "settings.phhide") || d) && wi(a, c, b)
                        }
                    })
                }), Zk = x(function (a) {
                    a = S(a);
                    a = Aq(a.search.substring(1));
                    a["_ym_status-check"] = a["_ym_status-check"] || "";
                    a._ym_lang = a._ym_lang || "ru";
                    return a
                }), zi = t(Zk, X("_ym_status-check"), Ga), Gs = t(Zk, X("_ym_lang")),
                yi = Ya(/^https:\/\/(yastatic\.net\/s3\/metrika|s3\.mds\.yandex\.net\/internal-metrika-betas|[\w-]+\.dev\.webvisor\.com|[\w-]+\.dev\.metrika\.yandex\.ru)\/(\w|-|\/|(\.)(?!\.))+\.js$/),
                lp = ["form", "button", "phone", "status"], dh = [], ip = x(function (a, c, b) {
                    z(t(Oc([a,
                        c, b]), ha), dh);
                    if (b.inline) {
                        c = xi(b);
                        var d = b.data;
                        b = b.id;
                        ti(a, c, void 0 === b ? "" : b, void 0 === d ? "" : d)
                    } else b.resource && yi(b.resource) && (a._ym__postMessageEvent = c, a._ym__inpageMode = b.inpageMode, a._ym__initMessage = b.initMessage, mp(a, b.resource))
                }, function (a, c, b) {
                    return b.id
                }), Hs = C("cs.init", function (a, c) {
                    var b, d = zi(a);
                    if (d && c.id === d && "0" === c.ba) {
                        var e = xi((b = {}, b.lang = Gs(a), b.fileId = "status", b));
                        V(a, E([a, e, "" + d], ti), 0, "cs")
                    }
                }), Is = C("suid.int", function (a, c) {
                    var b;
                    return b = {}, b.setUserID = function (d, e, f) {
                        if (wa(d) ||
                            qe(a, d)) {
                            var g = Ha(a, c);
                            d = Gc(["__ym", "user_id", d]);
                            g.params(d, e || B, f)
                        } else Sd(a, N(c)).error("Incorrect user ID")
                    }, b
                }), Kc = {position: "absolute"}, si = {position: "fixed"}, Tf = {borderRadius: "50%"},
                Js = pb("siteStatistics", function (a, c) {
                    if (!Yj(a)) return cc(a)(Sa(B, E([c, t(X("settings.sm"), ka(1), E([E([a, c.id], gp), B], Pe), ha)], ra)))
                }), Ks = C("up.int", function (a, c) {
                    var b;
                    return b = {}, b.userParams = D(a, "up.c", function (d, e, f) {
                        var g, h = Ha(a, c), k = Dd(a, N(c)).warn;
                        h ? La(d) ? (d = (g = {}, g.__ymu = d, g), (g = h.params) && g(d, e || B, f)) : k("Wrong user params") :
                            k("No counter instance found")
                    }), b
                }), Ls = /[\*\.\?\(\)]/g, Ms = x(function (a, c, b) {
                    try {
                        var d = b.replace("\\s", " ").replace(Ls, "");
                        Dd(a, "").warn('Function "' + d + '" has been overridden, this may cause issues with Metrika counter')
                    } catch (e) {
                    }
                }, yb), Ns = C("r.nn", function (a) {
                    Jf(a).isEnabled && xe(a, Ig, function (c) {
                        c.Aa.D(function (b) {
                            Ms(a, b[1], b[0]);
                            Ig.splice(100)
                        })
                    })
                }), Os = C("e.a.p", function (a, c) {
                    var b, d = Ha(a, c);
                    d = E([t(O, za(!0)), Z(Boolean, A(u(d, n), ["clickmap", "trackLinks", "accurateTrackBounce"]))], A);
                    c.Fh && d();
                    return b =
                        {}, b.enableAll = d, b
                }), Ps = u("add", pe), Qs = u("remove", pe), Rs = u("detail", pe), Ss = u("purchase", pe),
                Ts = "FB_IAB FBAV OKApp GSA/ yandex yango uber EatsKit YKeyboard iOSAppUslugi YangoEats PassportSDK".split(" "),
                jf = x(function (a) {
                    var c = Dk(a);
                    a = c.Mg;
                    if (!c.Nf) return !1;
                    c = qa("indexOf", a);
                    c = Ja(t(c, ka(-1), Tb), Ts);
                    var b = /CFNetwork\/[0-9][0-9.]*.*Darwin\/[0-9][0-9.]*/.test(a), d = /YaBrowser\/[\d.]+/.test(a),
                        e = /Mobile/.test(a);
                    return c || b || d && e || !/Safari/.test(a) && e
                }), Us = ["YangoEats"], $l = x(function (a) {
                    var c = gb(a);
                    if (!c) return !1;
                    c = qa("indexOf", c);
                    return Ja(t(c, ka(-1), Tb), Us) || pd(a)
                }), ep = /([0-9\\.]+) Safari/, Vs = /\sYptp\/\d\.(\d+)\s/, $k = x(function (a) {
                    var c;
                    a:{
                        if ((c = gb(a)) && (c = Vs.exec(c)) && 1 < c.length) {
                            c = Ga(c[1]);
                            break a
                        }
                        c = 0
                    }
                    return 50 <= c && 99 >= c || qf(a, 79) ? !1 : !Qd(a) || jf(a)
                }),
                al = "monospace;sans-serif;serif;Andale Mono;Arial;Arial Black;Arial Hebrew;Arial MT;Arial Narrow;Arial Rounded MT Bold;Arial Unicode MS;Bitstream Vera Sans Mono;Book Antiqua;Bookman Old Style;Calibri;Cambria;Cambria Math;Century;Century Gothic;Century Schoolbook;Comic Sans;Comic Sans MS;Consolas;Courier;Courier New;Garamond;Geneva;Georgia;Helvetica;Helvetica Neue;Impact;Lucida Bright;Lucida Calligraphy;Lucida Console;Lucida Fax;LUCIDA GRANDE;Lucida Handwriting;Lucida Sans;Lucida Sans Typewriter;Lucida Sans Unicode;Microsoft Sans Serif;Monaco;Monotype Corsiva;MS Gothic;MS Outlook;MS PGothic;MS Reference Sans Serif;MS Sans Serif;MS Serif;MYRIAD;MYRIAD PRO;Palatino;Palatino Linotype;Segoe Print;Segoe Script;Segoe UI;Segoe UI Light;Segoe UI Semibold;Segoe UI Symbol;Tahoma;Times;Times New Roman;Times New Roman PS;Trebuchet MS;Verdana;Wingdings;Wingdings 2;Wingdings 3".split(";"),
                Ws = x(function (a) {
                    a = ab(a)("canvas");
                    var c = n(a, "getContext");
                    if (!c) return null;
                    try {
                        var b = F(c, a)("2d");
                        b.font = "72px mmmmmmmmmmlli";
                        var d = b.measureText("mmmmmmmmmmlli").width;
                        return function (e) {
                            b.font = "72px " + e;
                            return b.measureText("mmmmmmmmmmlli").width === d
                        }
                    } catch (e) {
                        return null
                    }
                }), bl = Ma(String.prototype.repeat, "repeat"), Xs = bl ? function (a, c) {
                    return bl.call(a, c)
                } : bp, Nh = u(!0, function (a, c, b, d) {
                    b = c.length && (b - d.length) / c.length;
                    if (0 >= b) return d;
                    c = Xs(c, b);
                    return a ? c + d : d + c
                }), Re = [2277735313, 289559509], Se = [1291169091,
                    658871167], Ys = C("p.cd", function (a, c) {
                    if (od(a) || Ne(a)) {
                        var b = Ra(a);
                        if (ma(b.o("jn"))) {
                            b.C("jn", !1);
                            var d = a.nk || rd(a) ? function () {
                            } : /./, e = Sd(a, N(c));
                            d.toString = function () {
                                b.C("jn", !0);
                                return "Yandex.Metrika counter is initialized"
                            };
                            e.log("%c%s", "color: inherit", d)
                        }
                    }
                }), Zs = x(function (a) {
                    a = n(a, "navigator.plugins");
                    return !!(a && Pa(a) && Ja(t(X("name"), Ya(/Chrome PDF Viewer/)), a))
                }), Zo = {"*": "+", "-": "/", hk: "=", "+": "*", "/": "-", "=": "_"}, $s = x(function (a) {
                    return T(n(a, "yandex.getSiteUid")) ? a.yandex.getSiteUid() : null
                }),
                Vo = [["domainLookupEnd", "domainLookupStart"], ["connectEnd", "connectStart"], ["responseStart", "requestStart"], ["responseEnd", "responseStart"], ["fetchStart", "navigationStart"], ["redirectEnd", "redirectStart"], [function (a, c) {
                    return n(c, "redirectCount") || n(a, "navigation.redirectCount")
                }], ["domInteractive", "domLoading"], ["domContentLoadedEventEnd", "domContentLoadedEventStart"], ["domComplete", "navigationStart"], ["loadEventStart", "navigationStart"], ["loadEventEnd", "loadEventStart"], ["domContentLoadedEventStart",
                    "navigationStart"]], rb,
                Uo = [["domainLookupEnd", "domainLookupStart"], ["connectEnd", "connectStart"], ["responseStart", "requestStart"], ["responseEnd", "responseStart"], ["fetchStart"], ["redirectEnd", "redirectStart"], ["redirectCount"], ["domInteractive", "responseEnd"], ["domContentLoadedEventEnd", "domContentLoadedEventStart"], ["domComplete"], ["loadEventStart"], ["loadEventEnd", "loadEventStart"], ["domContentLoadedEventStart"]],
                qi = (rb = {}, rb.responseEnd = 1, rb.domInteractive = 1, rb.domContentLoadedEventStart = 1, rb.domContentLoadedEventEnd =
                    1, rb.domComplete = 1, rb.loadEventStart = 1, rb.loadEventEnd = 1, rb.unloadEventStart = 1, rb.unloadEventEnd = 1, rb.secureConnectionStart = 1, rb),
                Xo = x(vd), Ro = x(Ac), So = x(function (a) {
                    var c = n(a, "webkitRequestFileSystem");
                    if (T(c) && !od(a)) return (new I(F(c, a, 0, 0))).then(function () {
                        var d = n(a, "navigator.storage") || {};
                        return d.estimate ? d.estimate() : {}
                    }).then(function (d) {
                        return (d = d.quota) && 12E7 > d ? !0 : !1
                    })["catch"](u(!0, O));
                    if (Kd(a)) return c = n(a, "navigator.serviceWorker"), I.resolve(W(c));
                    c = n(a, "openDatabase");
                    if (rd(a) && T(c)) {
                        var b =
                            !1;
                        try {
                            c(null, null, null, null)
                        } catch (d) {
                            b = !0
                        }
                        return I.resolve(b)
                    }
                    return I.resolve(!n(a, "indexedDB") && (n(a, "PointerEvent") || n(a, "MSPointerEvent")))
                }), at = /(\?|&)turbo_uid=([\w\d]+)($|&)/, bt = x(function (a, c) {
                    var b = ac(a), d = S(a).search.match(at);
                    return d && 2 <= d.length ? (d = d[2], c.xb || b.C("turbo_uid", d), d) : (b = b.o("turbo_uid")) ? b : ""
                }), ct = C("pa.plgn", function (a, c) {
                    var b = de(a, c);
                    b && b.ca.D(["pluginInfo"], D(a, "c.plgn", function () {
                        var d = H(a);
                        d.C("cmc", d.o("cmc", 0) + 1);
                        return Pq(c, gf)
                    }))
                }), cl = bc.split("."), dt = cl.pop(),
                dl = J(".", cl), Sl = x(function (a) {
                    a = S(a).hostname.split(".");
                    return a[a.length - 1]
                }), sh = x(function (a) {
                    return -1 !== S(a).hostname.search(/(?:^|\.)(?:ya|yandex|beru|kinopoisk|edadeal)\.(?:\w+|com?\.\w+)$/)
                }),
                et = /^(.*\.)?((yandex(-team)?)\.(com?\.)?[a-z]+|(auto|kinopoisk|beru|bringly)\.ru|ya\.(ru|cc)|yadi\.sk|yastatic\.net|meteum\.(ai|es|io)|.*\.yandex|turbopages\.org|turbo\.site)$/,
                be = x(function (a) {
                    a = S(a).hostname;
                    var c = !1;
                    a && (c = -1 !== a.search(et));
                    return c
                }),
                ft = /^(.*\.)?((yandex(-team)?)\.(com?\.)?[a-z]+|(auto|kinopoisk|beru|bringly)\.ru|ya\.(ru|cc)|yadi\.sk|.*\.yandex|turbopages\.org|turbo\.site)$/,
                Go = x(function (a) {
                    a = S(a).hostname;
                    var c = !1;
                    a && (c = -1 !== a.search(ft));
                    return c
                }), el = sa.Za + "//" + bc + "/metrika", fl = el + "/metrika_match.html", sb, db,
                Tl = (sb = {}, sb.am = "com.am", sb.tr = "com.tr", sb.ge = "com.ge", sb.il = "co.il", sb["\u0440\u0444"] = "ru", sb["xn--p1ai"] = "ru", sb["\u0443\u043a\u0440"] = "ua", sb["xn--j1amh"] = "ua", sb["\u0431\u0435\u043b"] = "by", sb["xn--90ais"] = "by", sb),
                gl = {
                    "mc.edadeal.ru": /^([^/]+\.)?edadeal\.ru$/,
                    "mc.yandexsport.ru": /^([^/]+\.)?yandexsport\.ru$/,
                    "mc.kinopoisk.ru": /^([^/]+\.)?kinopoisk\.ru$/
                },
                Ul = (db = {}, db.ka = "ge", db.ro = "md", db.tg = "tj", db.tk = "tm", db.et = "ee", db.hy = "com.am", db.he = "co.li", db.ky = "kg", db.uk = "ua", db.be = "by", db.tr = "com.tr", db.kk = "kz", db),
                No = "ar:1:pv:1:v:" + sa.jb + ":vf:" + Sc.version, Oo = sa.Za + "//" + bc + "/watch/" + sa.Tg, hl = {},
                gt = C("exps.int", function (a, c) {
                    var b;
                    return b = {}, b.experiments = function (d, e, f) {
                        var g, h;
                        void 0 === e && (e = B);
                        if (d && 0 < d.length) {
                            var k = Ba(a, "e", c), l = Cd(c).url;
                            d = k({
                                H: Da((g = {}, g.ex = 1, g.ar = 1, g)),
                                G: (h = {}, h["page-url"] = l || S(a).href, h.exp = d, h)
                            }, c);
                            return Lc(a, "exps.s", d, e, f)
                        }
                    },
                        b
                }), kf = [], ht = C("p.fh", function (a, c) {
                    var b, d;
                    void 0 === c && (c = !0);
                    var e = Ra(a), f = fa(a), g = e.o("wasSynced"), h = {id: 3, ba: "0"};
                    return c && g && g.time + 864E5 > f(aa) ? I.resolve(g) : Ba(a, "f", h)({
                        H: Da((b = {}, b.pv = 1, b)),
                        G: (d = {}, d["page-url"] = S(a).href, d["page-ref"] = a.document.referrer, d)
                    }, h).then(function (k) {
                        var l;
                        k = (l = {}, l.time = f(aa), l.params = n(k, "settings"), l.bkParams = n(k, "userData"), l);
                        e.C("wasSynced", k);
                        return k
                    })["catch"](D(a, "f.h"))
                }), it = la(function (a, c) {
                    0 === parseFloat(n(c, "settings.c_recp")) && (a.ke.C("ymoo" + a.da,
                        a.Eg(lb)), a.Cd && a.Cd.destruct && a.Cd.destruct())
                }), oi = t(X("settings.pcs"), ka("1")),
                Do = [[["'(-$&$&$'", 30102, 0], ["'(-$&$&$'", 29009, 0]], [["oWdZ[nc[jh_YW$Yec", 30103, 1], ["oWdZ[nc[jh_YW$Yec", 29010, 1]]],
                Eo = [[["oWdZ[nc[jh_YW$Yec", 30103, 1]], [["oWdZ[nc[jh_YW$Yec", 29010, 1]]],
                pi = {G: {t: 'UV|L7,!"T[rwe&D_>ZIb\\aW#98Y.PC6k'}}, ni = {id: 42822899, ba: "0"}, $e,
                Ko = ($e = {}, $e.s = "p", $e["8"] = "i", $e), Ho = pb("csp", function (a, c) {
                    return Ba(a, "s", c)({}, ["https://ymetrica1.com/watch/3/1"])
                }), eh = "et w v z i u vf".split(" "), wo = {
                    L: "stamp",
                    M: "frameId",
                    aa: "meta",
                    hf: "base",
                    Jf: "hasBase",
                    ef: "address",
                    Jg: "ua",
                    ze: "prev",
                    Vf: "namespace",
                    Sc: "keystrokes",
                    Of: "isMeta",
                    Tc: "modifier",
                    Bb: "pageWidth",
                    Ab: "pageHeight",
                    yg: "startNode",
                    pf: "endNode",
                    Qg: "zoomFrom",
                    Sg: "zoomTo",
                    level: "level",
                    duration: "duration",
                    Ha: "i",
                    Wc: "o",
                    n: "n",
                    r: "r",
                    Ac: "ct",
                    Ob: "at",
                    Wf: "nm",
                    Xf: "ns",
                    ue: "pa",
                    xe: "pr",
                    ne: "nx",
                    Qa: "h",
                    Oa: "changes",
                    cf: "a",
                    gf: "b",
                    wd: "c",
                    te: "op"
                }, xo = ["attributes", "attrs"], af = function () {
                    function a(c) {
                        this.l = c
                    }

                    a.prototype.Ca = function (c) {
                        var b = me(c);
                        c = A(F(this.Ua, this),
                            b);
                        return yf(mb(this.l, A(function (d, e) {
                            var f;
                            return y({}, b[e], (f = {}, f.data = d, f))
                        }, c)))
                    };
                    a.prototype.Ua = function (c) {
                        var b = c.data;
                        "string" !== typeof b && (b = mb(this.l, me(c.data)));
                        return b
                    };
                    a.prototype.$a = function (c) {
                        return encodeURIComponent(c).length
                    };
                    a.prototype.kd = function (c, b) {
                        for (var d = Math.ceil(c.length / b), e = [], f = 0; f < b; f += 1) e.push(c.slice(f * d, d * (f + 1)));
                        return e
                    };
                    a.prototype.isEnabled = function () {
                        return !!this.l.JSON
                    };
                    return a
                }(), uo = x(function (a) {
                    function c(f, g, h, k) {
                        d[0] = g;
                        h[k] = e[3];
                        h[k + 1] = e[2];
                        h[k +
                        2] = e[1];
                        h[k + 3] = e[0]
                    }

                    function b(f, g, h, k) {
                        d[0] = g;
                        h[k] = e[0];
                        h[k + 1] = e[1];
                        h[k + 2] = e[2];
                        h[k + 3] = e[3]
                    }

                    if ("undefined" === typeof a.Float32Array || "undefined" === typeof a.Uint8Array) return vo;
                    var d = new Float32Array([-0]), e = new Uint8Array(d.buffer);
                    return 128 === e[3] ? b : c
                }), qo = ki(!1), po = ki(!0), jt = {
                    Jj: "topics",
                    pj: "publicationDate",
                    mj: "pageUrlCanonical",
                    Uj: "updateDate",
                    jh: "authors",
                    gh: "articleInfo",
                    vj: "rubric",
                    qj: "publishersHeader",
                    Ra: "involvedTime",
                    lj: "pageTitle",
                    Si: "maxScrolled",
                    kf: "chars",
                    hh: "articleMeta"
                }, kt = function () {
                    function a(c) {
                        this.l =
                            c;
                        this.Xb = t(pa, xd(function (b, d) {
                            b[d[1]] = d[0];
                            return b
                        }, {}))(jt)
                    }

                    a.prototype.Cg = function (c) {
                        var b, d = this.ye(c.data);
                        return {L: fa(this.l)(Ag), data: (b = {}, b[this.Xb[c.type]] = d, b)}
                    };
                    a.prototype.ye = function (c) {
                        var b = this;
                        return ca(c) ? A(F(this.ye, this), c) : La(c) ? t(pa, xd(function (d, e) {
                            var f = e[0];
                            d[b.Xb[f] || f] = b.ye(e[1]);
                            return d
                        }, {}))(c) : c
                    };
                    a.prototype.Ua = function (c) {
                        return ke(this.l, Of, this.Cg(c))(Me(B))
                    };
                    a.prototype.$a = function (c) {
                        return c[0]
                    };
                    a.prototype.kd = function (c) {
                        return [c]
                    };
                    a.prototype.Ca = function (c) {
                        var b =
                            this;
                        c = ke(this.l, di, {buffer: A(F(this.Cg, this), c)});
                        return kc(this.l, c, 20, Le)(Xe(function (d) {
                            d = Lf(b.l, d.slice(-4));
                            return kc(b.l, d, 20, Le)
                        }))(Vg(function (d) {
                            return d[d.length - 1]
                        }))
                    };
                    a.prototype.isEnabled = function () {
                        return ci(this.l)
                    };
                    return a
                }(),
                il = "resize scroll mousemove mousedown click windowfocus keydown orientationchange change focus touchmove touchstart".split(" "),
                lt = "id pageTitle stamp chars authors updateDate publicationDate pageUrlCanonical topics rubric".split(" "),
                mt = function () {
                    function a(c,
                               b, d, e, f) {
                        var g = this;
                        this.Qc = !1;
                        this.aa = {};
                        this.scroll = {x: 0, y: 0};
                        this.Ra = this.Tf = 0;
                        this.le = this.bg = "";
                        this.ha = [];
                        this.Ke = this.Wa = 0;
                        this.Jb = {Qa: 0, pd: 0};
                        this.buffer = [];
                        this.Ug = lt;
                        this.flush = function () {
                            g.Ke = V(g.l, g.flush, 2500);
                            var h = g.Qd();
                            if (g.buffer.length || h) {
                                var k = Ed(g.buffer);
                                h && k.push(h);
                                g.bg = g.le;
                                g.ra.Ca(k)(Sa(D(g.l, "p.b.st"), function (l) {
                                    l && g.ec(l)
                                }))
                            }
                        };
                        this.ec = e;
                        this.ra = d;
                        this.nc = F(this.nc, this);
                        this.Qd = F(this.Qd, this);
                        this.flush = F(this.flush, this);
                        this.l = c;
                        this.da = f;
                        this.Eb = b;
                        this.ee = "pai" + b.id;
                        this.Ub();
                        this.rf = ia(this.l);
                        this.time = fa(this.l);
                        this.Kg();
                        this.Td = H(this.l)
                    }

                    a.prototype.start = function () {
                        var c = this;
                        this.Ke = V(this.l, this.flush, 2500);
                        if (!this.Qc) {
                            this.Dj();
                            var b = this.Td.o(this.ee, []), d = !b.length;
                            b.push(F(this.Oi, this));
                            this.Td.Va(this.ee, b);
                            d && this.kg();
                            var e = function (f, g) {
                                return (f.He || 0) <= (g.He || 0) ? g : f
                            };
                            ia(this.l).D(this.l, ["click"], function (f) {
                                if (c.ha.length) {
                                    f = Xi(f);
                                    var g = S(c.l).hostname, h;
                                    if (h = f) h = we(f.hostname) === we(g);
                                    h && (f = M(e, c.ha[0], c.ha).id, g = Lb(c.l), Uc(c.l, c.da.split(":")[0]).C("pai",
                                        f + "-" + g))
                                }
                            });
                            this.nc({type: "page", target: this.l})
                        }
                    };
                    a.prototype.stop = function () {
                        this.Sj();
                        this.Qc = !0;
                        this.flush();
                        na(this.l, this.Ke)
                    };
                    a.prototype.Qf = function (c) {
                        return dc("html", this.l, c) !== this.l.document.documentElement
                    };
                    a.prototype.kg = function () {
                        var c = this;
                        D(this.l, "p.ic" + this.Eb.id, function () {
                            if (!c.Qc) {
                                var b = c.Td.o(c.ee), d = c.Eb.Oh();
                                z(function (e) {
                                    var f = A(function (g) {
                                        return y({}, g)
                                    }, d);
                                    T(e) && e(f)
                                }, b);
                                c.Wa = V(c.l, F(c.kg, c), 1E3, "p")
                            }
                        })()
                    };
                    a.prototype.Oi = function (c) {
                        this.Qc || (this.Tj(c), this.Vj(),
                            this.qh())
                    };
                    a.prototype.nc = function (c) {
                        var b = this;
                        D(this.l, "p.ec." + this.Eb.id, function () {
                            try {
                                var d = c.type;
                                var e = c.target
                            } catch (l) {
                                return
                            }
                            var f = "page" === d;
                            if ("scroll" === d || f) {
                                var g = [b.l, b.l.document, b.l.document.documentElement, yc(b.l)];
                                K(e, g) && b.Ub()
                            }
                            ("resize" === d || f) && b.Kg();
                            d = b.time(aa);
                            var h = Math.min(d - b.Tf, 5E3);
                            b.Ra += Math.round(h);
                            b.Tf = d;
                            if (b.aa && b.scroll && b.Jb) {
                                var k = b.Jb.Qa * b.Jb.pd;
                                b.ha = A(function (l) {
                                    var m = y({}, l), p = b.aa[m.id], q = Ic(l.vc);
                                    if (!p || b.Qf(m.element) || !q) return m;
                                    l = b.l.Math;
                                    p = l.max((b.scroll.y +
                                        b.Jb.Qa - p.y) / p.height, 0);
                                    var r = q.height * q.width;
                                    q = Cj(b.l, q, b.Jb);
                                    m.He = q / k;
                                    m.visibility = q / r;
                                    if (.9 <= m.visibility || .1 <= m.He) m.involvedTime += h;
                                    m.maxScrolled = l.round(1E4 * p) / 1E4;
                                    return m
                                }, b.ha);
                                ib(b.l, {name: "publishers", da: b.da, data: {Ra: b.Ra, ha: b.ha}})
                            }
                        })()
                    };
                    a.prototype.Tj = function (c) {
                        var b = A(function (d) {
                            return d.id
                        }, this.ha);
                        this.ha = this.ha.concat(Z(function (d) {
                            return !K(d.id, b)
                        }, c))
                    };
                    a.prototype.Kg = function () {
                        var c = Je(this.l) || Qc(this.l);
                        this.Jb = {pd: c[0], Qa: c[1]}
                    };
                    a.prototype.Vj = function () {
                        var c = this;
                        D(this.l, "p.um." + this.Eb.id, function () {
                            var b = [];
                            c.Ub();
                            c.aa = M(function (d, e) {
                                var f;
                                if (c.Qf(e.element)) b.push(e), delete d[e.id]; else {
                                    var g = (f = {}, f.id = e.id, f.involvedTime = Math.max(e.involvedTime, 0), f.maxScrolled = e.maxScrolled || 0, f.chars = e.update ? e.update("chars") || 0 : 0, f);
                                    e.vc && (f = Ic(e.vc)) && (g.x = Math.max(Math.round(f.left) + c.scroll.x, 0), g.y = Math.max(Math.round(f.top) + c.scroll.y, 0), g.width = Math.round(f.width), g.height = Math.round(f.height));
                                    d[e.id] = g
                                }
                                return d
                            }, {}, c.ha);
                            z(function (d) {
                                d = ye(c.l)(d, c.ha);
                                c.ha.splice(d, 1)
                            }, b)
                        })()
                    };
                    a.prototype.Qd = function () {
                        var c, b, d = A(u(this.aa, n), da(this.aa));
                        return d.length && (this.le = mb(this.l, d), this.bg !== this.le) ? (c = {}, c.type = "publishersHeader", c.data = (b = {}, b.articleMeta = d || [], b.involvedTime = this.Ra, b), c) : null
                    };
                    a.prototype.qh = function () {
                        var c = this;
                        if (this.ha.length) {
                            var b = A(function (d) {
                                var e, f = M(function (g, h) {
                                    d[h] && (g[h] = d[h]);
                                    return g
                                }, {}, c.Ug);
                                d.ug = !0;
                                return e = {}, e.type = "articleInfo", e.stamp = f.stamp, e.data = f, e
                            }, Z(function (d) {
                                return !d.ug
                            }, this.ha));
                            b.length &&
                            (this.buffer = this.buffer.concat(b), Db(this.l, this.da, "Publisher content info found: ", b))
                        }
                    };
                    a.prototype.Dj = function () {
                        this.rf.D(this.l, il, this.nc)
                    };
                    a.prototype.Sj = function () {
                        this.rf.kc(this.l, il, this.nc)
                    };
                    a.prototype.Ub = function () {
                        this.scroll = {
                            x: this.l.pageXOffset || n(this.l, "document.documentElement.scrollLeft") || 0,
                            y: this.l.pageYOffset || n(this.l, "document.documentElement.scrollLeft") || 0
                        }
                    };
                    return a
                }(), Ud, fh = (Ud = {}, Ud[1] = 500, Ud[2] = 500, Ud[3] = 0, Ud), gh = function () {
                    function a(c, b) {
                        var d, e = this;
                        this.id = "a";
                        this.Yd = !1;
                        this.Sb = {};
                        this.Fb = {
                            "schema.org": "Article NewsArticle Movie BlogPosting Review Recipe Answer".split(" "),
                            Zf: ["article"]
                        };
                        this.Ve = (d = {}, d.Answer = 3, d.Review = 2, d);
                        this.xh = x(function (f, g) {
                            Db(e.l, e.da, "Warning: content has only " + g.chars + " chars. Required " + fh[g.type], g)
                        });
                        this.l = c;
                        this.root = Yb(c);
                        this.da = b
                    }

                    a.prototype.Pa = function (c) {
                        return c.element
                    };
                    a.prototype.Cf = function (c, b) {
                        var d = this, e;
                        D(this.l, "P.s." + b, function () {
                            e = d.Sb[b].call(d, c)
                        })();
                        return e
                    };
                    a.prototype.oj = function (c) {
                        var b = y({},
                            c);
                        this.Yd && !b.id && K(c.type, [3, 2]) && (c = J(", ", A(X("name"), b.authors || [])), b.pageTitle = c + ": " + b.pageTitle);
                        b.pageTitle || (b.pageTitle = this.ni(b.vc));
                        b.pageUrlCanonical || (c = b.id, b.pageUrlCanonical = ("string" !== typeof c ? 0 : /^(https?:)\/\//.test(c)) ? b.id : this.ki());
                        b.id || (b.id = b.pageTitle || b.pageUrlCanonical);
                        return b
                    };
                    a.prototype.Ga = function (c) {
                        var b = this, d = {}, e = this.Pa(c);
                        if (!e) return null;
                        d.type = c.type;
                        z(function (g) {
                            d[g] = b.Cf(c, g)
                        }, da(this.Sb));
                        var f = fa(this.l);
                        d.stamp = f(Ag);
                        d.element = c.element;
                        d.vc =
                            e;
                        d = this.oj(d);
                        d.id = d.id ? oc(d.id) : 1;
                        d.update = function (g) {
                            return b.Pa(c) ? b.Cf(c, g) : void 0
                        };
                        return d
                    };
                    a.prototype.ni = function (c) {
                        for (var b = 1; 5 >= b; b += 1) {
                            var d = Za(qc("h" + b, c));
                            if (d) return d
                        }
                    };
                    a.prototype.ki = function () {
                        var c = qc('[rel="canonical"]', this.root);
                        if (c) return c.href
                    };
                    a.prototype.Hf = function () {
                        return 1
                    };
                    a.prototype.Gc = function () {
                        return []
                    };
                    a.prototype.Oh = function () {
                        var c = this, b = this.Gc(), d = 1;
                        return M(function (e, f) {
                            var g = c.Ga({element: f, type: c.Hf(f)}) || [];
                            ca(g) || (g = [g]);
                            g = M(function (h, k) {
                                var l =
                                    h.values, m = h.Kf;
                                k && k.chars > fh[k.type] && !K(k.id, m) ? (l.push(k), m.push(k.id)) : k && k.chars <= fh[k.type] && c.xh(k.id, k);
                                return {values: l, Kf: m}
                            }, {values: [], Kf: A(X("id"), e)}, g).values;
                            return e.concat(A(function (h) {
                                var k;
                                h = y((k = {index: d, ug: !1}, k.involvedTime = 0, k), h);
                                d += 1;
                                return h
                            }, g))
                        }, [], b)
                    };
                    return a
                }(), jl = function (a) {
                    function c() {
                        var b, d = null !== a && a.apply(this, arguments) || this;
                        d.id = "j";
                        d.Yd = !0;
                        d.lf = J(",", ['script[type="application/ld+json"]', 'script[type="application/json+ld"]', 'script[type="ld+json"]', 'script[type="json+ld"]']);
                        d.Sb = (b = {}, b.id = function (e) {
                            var f = e.data["@id"];
                            e = e.data.mainEntity || e.data.mainEntityOfPage;
                            !f && e && (f = e["@id"]);
                            return f
                        }, b.chars = function (e) {
                            return "string" === typeof e.data.text ? e.data.text.length : Sb(this.Pa(e)).length
                        }, b.authors = function (e) {
                            var f = [];
                            f = f.concat(this.Fc(e.data, "author"));
                            f = f.concat(this.Fc(e.data.mainEntity, "author"));
                            return f.concat(this.Fc(e.data.mainEntityOfPage, "author"))
                        }, b.pageTitle = function (e) {
                            var f = e.data.headline || "";
                            e.data.jk && (f += " " + e.data.alternativeHeadline);
                            "" === f &&
                            (e.data.name ? f = e.data.name : e.data.itemReviewed && (f = e.data.itemReviewed));
                            3 === e.type && e.data.parentItem && (f = e.data.parentItem.text);
                            return f
                        }, b.updateDate = function (e) {
                            return e.data.dateModified || ""
                        }, b.publicationDate = function (e) {
                            return e.data.datePublished || ""
                        }, b.pageUrlCanonical = function (e) {
                            return e.data.url
                        }, b.topics = function (e) {
                            return this.Fc(e.data, "about", ["name", "alternateName"])
                        }, b.rubric = function (e) {
                            var f = this, g = this.Pa(e);
                            e = Z(Boolean, A(function (h) {
                                if (h = tb(f.l, Sb(h))) {
                                    var k = f.Df(h);
                                    if (k) return M(function (l,
                                                              m) {
                                        return l ? l : "BreadcrumbList" === m["@type"] ? m : l
                                    }, null, k);
                                    if ("BreadcrumbList" === h["@type"]) return h
                                }
                                return null
                            }, [e.element].concat(xb(this.lf, document.body === g ? document.documentElement : g))));
                            return e.length && (e = e[0].itemListElement, ca(e)) ? Z(Boolean, A(function (h) {
                                return La(h) && h.item && La(h.item) && !f.l.isNaN(h.position) ? {
                                    name: h.item.name || h.name,
                                    position: h.position
                                } : null
                            }, e)) : []
                        }, b);
                        return d
                    }

                    Ka(c, a);
                    c.prototype.Fc = function (b, d, e) {
                        void 0 === e && (e = ["name"]);
                        if (!b || !b[d]) return [];
                        b = ca(b[d]) ? b[d] : [b[d]];
                        b = Z(O, A(function (f) {
                            return f ? "string" === typeof f ? f : M(function (g, h) {
                                return g || "" + f[h]
                            }, "", e) : null
                        }, b));
                        return A(function (f) {
                            var g;
                            return g = {}, g.name = f, g
                        }, b)
                    };
                    c.prototype.Pa = function (b) {
                        var d = b.element, e = b.data["@id"], f = b.data.url;
                        b = null;
                        f && "string" === typeof f && (b = this.uf(f));
                        !b && e && "string" === typeof e && (b = this.uf(e));
                        b || (b = e = d.parentNode, !dc("head", this.l, d) && e && 0 !== Sb(e).length) || (b = this.l.document.body);
                        return b
                    };
                    c.prototype.uf = function (b) {
                        try {
                            var d = Hc(this.l, b).hash;
                            if (d) {
                                var e = qc(d, this.l.document.body);
                                if (e) return e
                            }
                        } catch (f) {
                        }
                        return null
                    };
                    c.prototype.ve = function (b) {
                        return this.Ve[b["@type"]] || 1
                    };
                    c.prototype.Ga = function (b) {
                        var d = this, e = b.element;
                        if (!b.data && (b.data = tb(this.l, Sb(e)), !b.data || !/schema\.org/.test(b.data["@context"]) && !ca(b.data))) return null;
                        var f = this.Df(b.data);
                        if (f) return A(function (h) {
                            if (!K(h["@type"], d.Fb["schema.org"])) return null;
                            h = {element: e, data: h, type: d.ve(h)};
                            return a.prototype.Ga.call(d, h)
                        }, f);
                        if ("QAPage" === b.data["@type"]) {
                            var g = b.data.mainEntity || b.data.mainEntityOfPage;
                            if (!g) return null
                        }
                        "Question" === b.data["@type"] && (g = b.data);
                        return g ? (b = mc(u(g, n), ["acceptedAnswer", "suggestedAnswer"]), A(function (h) {
                            var k;
                            if (!h || !K(h["@type"], d.Fb["schema.org"])) return null;
                            h = {element: e, type: d.ve(h), data: y((k = {}, k.parentItem = g, k), h)};
                            return a.prototype.Ga.call(d, h)
                        }, b)) : K(b.data["@type"], this.Fb["schema.org"]) ? a.prototype.Ga.call(this, y(b, {type: this.ve(b.data)})) : null
                    };
                    c.prototype.Gc = function () {
                        return xb(this.lf, this.root)
                    };
                    c.prototype.Df = function (b) {
                        return ca(b) && b || b && ca(b["@graph"]) &&
                            b["@graph"]
                    };
                    return c
                }(gh), hh = function (a) {
                    function c() {
                        var b, d = null !== a && a.apply(this, arguments) || this;
                        d.id = "s";
                        d.Yd = !0;
                        d.Rj = qa("exec", new RegExp("schema.org\\/(" + J("|", da(d.Ve)) + ")$"));
                        d.Sb = (b = {}, b.id = function (e) {
                            e = e.element;
                            var f = cb(this.l, e, "identifier");
                            return f ? Za(f) : (f = cb(this.l, e, "mainEntityOfPage")) && f.getAttribute("itemid") ? f.getAttribute("itemid") : null
                        }, b.chars = function (e) {
                            var f = 0;
                            e = e.element;
                            for (var g = ["articleBody", "reviewBody", "recipeInstructions", "description", "text"], h = 0; h < g.length; h +=
                                1) {
                                var k = cb(this.l, e, g[h]);
                                if (k) {
                                    f = Za(k).length;
                                    break
                                }
                            }
                            e = Sb(e);
                            0 === f && e && (f += e.length);
                            return f
                        }, b.topics = function (e) {
                            var f = this, g = Pd(this.l, e.element, "about");
                            return A(function (h) {
                                var k = {name: Za(h)};
                                if (g = cb(f.l, h, "name")) k.name = Za(g);
                                return k
                            }, g)
                        }, b.rubric = function (e) {
                            var f = this;
                            (e = qc('[itemtype$="schema.org/BreadcrumbList"]', e.element)) || (e = qc('[itemtype$="schema.org/BreadcrumbList"]', this.root));
                            return e ? A(function (g) {
                                return {name: Za(cb(f.l, g, "name")), position: Za(cb(f.l, g, "position"))}
                            }, Pd(this.l,
                                e, "itemListElement")) : []
                        }, b.updateDate = function (e) {
                            return (e = cb(this.l, e.element, "dateModified")) ? ok(e) : ""
                        }, b.publicationDate = function (e) {
                            return (e = cb(this.l, e.element, "datePublished")) ? ok(e) : ""
                        }, b.pageUrlCanonical = function (e) {
                            e = Pd(this.l, e.element, "url");
                            if (e.length) {
                                var f = e[0];
                                return f.href ? f.href : Za(e)
                            }
                            return null
                        }, b.pageTitle = function (e) {
                            var f = "", g = e.element, h = cb(this.l, g, "headline");
                            h && (f += Za(h));
                            (h = cb(this.l, g, "alternativeHeadline")) && (f += " " + Za(h));
                            "" === f && ((h = cb(this.l, g, "name")) || (h = cb(this.l,
                                g, "itemReviewed")), h && (f += Za(h)));
                            3 === e.type && (e = dc('[itemtype$="schema.org/Question"]', this.l, g)) && (e = cb(this.l, e, "text")) && (f += Za(e));
                            return f
                        }, b.authors = function (e) {
                            var f = this;
                            e = Pd(this.l, e.element, "author");
                            return A(function (g) {
                                var h, k = (h = {}, h.name = "", h);
                                /.+schema.org\/(Person|Organization)/.test(g.getAttribute("itemtype") || "") && (h = cb(f.l, g, "name")) && (k.name = Za(h));
                                k.name || (k.name = g.getAttribute("content") || Sb(g) || g.getAttribute("href"));
                                return k
                            }, e)
                        }, b);
                        return d
                    }

                    Ka(c, a);
                    c.prototype.Hf = function (b) {
                        b =
                            b.getAttribute("itemtype") || "";
                        return (b = this.Rj(b)) ? this.Ve[b[1]] : 1
                    };
                    c.prototype.Ga = function (b) {
                        return b.element && Sb(b.element).length ? a.prototype.Ga.call(this, b) : null
                    };
                    c.prototype.Gc = function () {
                        var b = J(",", A(function (d) {
                            return '[itemtype$="schema.org/' + d + '"]'
                        }, this.Fb["schema.org"]));
                        return xb(b, this.root)
                    };
                    return c
                }(gh), kl = function (a) {
                    function c(b, d) {
                        var e, f = a.call(this, b, d) || this;
                        f.id = "o";
                        f.Sb = (e = {}, e.chars = function (g) {
                            g = this.Pa(g);
                            return Sb(g).length
                        }, e.authors = function (g) {
                            return this.Md(g.data.author)
                        },
                            e.pageTitle = function (g) {
                                return this.Kc(g.data.title) || ""
                            }, e.updateDate = function (g) {
                            return this.Kc(g.data.modified_time)
                        }, e.publicationDate = function (g) {
                            return this.Kc(g.data.published_time)
                        }, e.pageUrlCanonical = function (g) {
                            return this.Kc(g.data.url)
                        }, e.rubric = function (g) {
                            return this.Md(g.data.section)
                        }, e.topics = function (g) {
                            return this.Md(g.data.tag)
                        }, e);
                        f.Gh = new RegExp("^(og:)?((" + J("|", f.Fb.Zf) + "):)?");
                        return f
                    }

                    Ka(c, a);
                    c.prototype.Md = function (b) {
                        var d;
                        return b ? ca(b) ? A(function (e) {
                            var f;
                            return f = {},
                                f.name = e, f
                        }, b) : [(d = {}, d.name = b, d)] : []
                    };
                    c.prototype.Kc = function (b) {
                        return ca(b) ? b.length ? b[0] : null : b
                    };
                    c.prototype.Gc = function () {
                        var b = xb('meta[property="og:type"]', this.l.document.body);
                        return [this.l.document.head].concat(b)
                    };
                    c.prototype.ai = function (b) {
                        var d = this, e = b.element, f = {}, g = this.Pa(b);
                        e = xb("meta[property]", e === this.l.document.head ? e : g);
                        if (e.length) z(function (h) {
                            try {
                                if (h.parentNode === g || h.parentNode === d.l.document.head) {
                                    var k = h.getAttribute("property").replace(d.Gh, ""), l = Za(h);
                                    f[k] ? ca(f[k]) ?
                                        f[k].push(l) : f[k] = [f[k], l] : f[k] = l
                                }
                            } catch (m) {
                                oe(d.l, "og.ed", m)
                            }
                        }, e); else return null;
                        return K(f.type, this.Fb.Zf) ? y(b, {data: f}) : null
                    };
                    c.prototype.Pa = function (b) {
                        b = b.element;
                        var d = this.l.document;
                        return b === d.head ? d.body : b.parentNode
                    };
                    c.prototype.Ga = function (b) {
                        return (b = this.ai(b)) ? a.prototype.Ga.call(this, b) : null
                    };
                    return c
                }(gh), Vd = {};
            jl && (Vd.json_ld = jl);
            hh && (Vd.schema = hh, Vd.microdata = hh);
            kl && (Vd.opengraph = kl);
            var nt = C("p.p", function (a, c) {
                    function b(l) {
                        var m = y({}, k);
                        m.Y.fa = l;
                        return e(m, c)["catch"](D(a,
                            "s.ww.p"))
                    }

                    if (!Aa("querySelectorAll", a.document.querySelectorAll)) return I.resolve();
                    var d = [new af(a)];
                    d.unshift(new kt(a));
                    var e = Ba(a, "p", c), f = bb(function (l) {
                        return l.isEnabled()
                    }, d);
                    d = Da();
                    var g = Uc(a, c.id), h = g.o("pai");
                    h && (g.Rb("pai"), d.C("pai", h));
                    var k = {G: {}, H: d, Ta: {Zd: !(f instanceof af)}, Y: {}};
                    return ra(c, D(a, "ps.s", function (l) {
                        if (l = n(l, "settings.publisher.schema")) {
                            Xj(c) && (l = "microdata");
                            var m = Vd[l];
                            if (m && f) {
                                var p = N(c);
                                m = new m(a, p);
                                (new mt(a, m, f, b, p)).start();
                                Db(a, p, 'Publishers analytics schema "' +
                                    l + '"')
                            }
                        }
                    }))
                }), ot = function () {
                    function a(c, b) {
                        this.l = c;
                        this.Zh = b
                    }

                    a.prototype.Ca = function (c) {
                        return yf(mc(F(this.Ua, this), c))
                    };
                    a.prototype.Ua = function (c, b) {
                        var d = this, e = [], f = this.Zh(this.l, b && b.type, c.type);
                        f && (e = mc(function (g) {
                            return g({l: d.l, sa: c}) || []
                        }, f));
                        return e
                    };
                    a.prototype.$a = function (c) {
                        return c.length
                    };
                    a.prototype.kd = function (c) {
                        return [c]
                    };
                    a.prototype.isEnabled = function () {
                        return !0
                    };
                    return a
                }(), ll = function () {
                    function a(c, b, d) {
                        this.nf = 0;
                        this.we = 1;
                        this.rd = 5E3;
                        this.l = c;
                        this.ra = b;
                        this.ec = d
                    }

                    a.prototype.jd =
                        function () {
                            this.nf = V(this.l, t(F(this.flush, this), F(this.jd, this)), this.rd, "b.f")
                        };
                    a.prototype.send = function (c, b) {
                        var d = this.ec(c, b || [], this.we);
                        this.we += 1;
                        return d
                    };
                    a.prototype.push = function () {
                    };
                    a.prototype.flush = function () {
                    };
                    return a
                }(), kn = function (a) {
                    function c(b, d, e) {
                        b = a.call(this, b, d, e) || this;
                        b.buffer = [];
                        b.Vg = 7500;
                        b.rd = 3E4;
                        b.jd();
                        return b
                    }

                    Ka(c, a);
                    c.prototype.push = function (b, d) {
                        var e = this.ra.Ua(b, d);
                        Oa(this.buffer, e);
                        this.ra.$a(this.buffer) > this.Vg && this.flush()
                    };
                    c.prototype.flush = function () {
                        var b =
                            this.buffer;
                        b.length && (this.send(b), this.buffer = [])
                    };
                    return c
                }(ll), An = /opera mini/i, Yh = ["phone", "email"],
                ml = "first(-|\\.|_|\\s){0,2}name last(-|\\.|_|\\s){0,2}name zip postal address passport (bank|credit)(-|\\.|_|\\s){0,2}card card(-|\\.|_|\\s){0,2}number card(-|\\.|_|\\s){0,2}holder cvv card(-|\\.|_|\\s){0,2}exp card(-|\\.|_|\\s){0,2}name card.*month card.*year card.*month card.*year password birth(-|\\.|_|\\s){0,2}(day|date) second(-|\\.|_|\\s){0,2}name third(-|\\.|_|\\s){0,2}name patronymic middle(-|\\.|_|\\s){0,2}name birth(-|\\.|_|\\s){0,2}place house street city flat state contact.*".split(" "),
                xn = /^[\w\u0410-\u042f\u0430-\u044f]$/, yn = [65, 90], zn = [97, 122],
                vn = "color radio checkbox date datetime-local email month number password range search tel text time url week".split(" "),
                tn = new RegExp("(" + J("|", ml) + ")", "i"), sn = new RegExp("(" + J("|", Yh) + ")", "i"),
                pk = ["password", "passwd", "pswd"],
                un = new RegExp("(" + J("|", ml.concat("\u0438\u043c\u044f \u0444\u0430\u043c\u0438\u043b\u0438\u044f \u043e\u0442\u0447\u0435\u0441\u0442\u0432\u043e \u0438\u043d\u0434\u0435\u043a\u0441 \u0442\u0435\u043b\u0435\u0444\u043e\u043d \u0430\u0434\u0440\u0435\u0441 \u043f\u0430\u0441\u043f\u043e\u0440\u0442 \u043d\u043e\u043c\u0435\u0440(-|\\.|_|\\s){0,2}\u043a\u0430\u0440\u0442\u044b \u0434\u0430\u0442\u0430(-|\\.|_|\\s){0,2}\u0440\u043e\u0436\u0434\u0435\u043d\u0438\u044f \u0434\u043e\u043c \u0443\u043b\u0438\u0446\u0430 \u043a\u0432\u0430\u0440\u0442\u0438\u0440\u0430 \u0433\u043e\u0440\u043e\u0434 \u043e\u0431\u043b\u0430\u0441\u0442\u044c".split(" "))) +
                    ")", "i"), Wa = "metrikaId_" + Math.random(), sc = {Bd: 0}, or = x(function () {
                    var a;
                    return a = {}, a.A = 1, a.ABBR = 2, a.ACRONYM = 3, a.ADDRESS = 4, a.APPLET = 5, a.AREA = 6, a.B = 7, a.BASE = 8, a.BASEFONT = 9, a.BDO = 10, a.BIG = 11, a.BLOCKQUOTE = 12, a.BODY = 13, a.BR = 14, a.BUTTON = 15, a.CAPTION = 16, a.CENTER = 17, a.CITE = 18, a.CODE = 19, a.COL = 20, a.COLGROUP = 21, a.DD = 22, a.DEL = 23, a.DFN = 24, a.DIR = 25, a.DIV = 26, a.DL = 27, a.DT = 28, a.EM = 29, a.FIELDSET = 30, a.FONT = 31, a.FORM = 32, a.FRAME = 33, a.FRAMESET = 34, a.H1 = 35, a.H2 = 36, a.H3 = 37, a.H4 = 38, a.H5 = 39, a.H6 = 40, a.HEAD = 41, a.HR = 42, a.HTML =
                        43, a.I = 44, a.IFRAME = 45, a.IMG = 46, a.INPUT = 47, a.INS = 48, a.ISINDEX = 49, a.KBD = 50, a.LABEL = 51, a.LEGEND = 52, a.LI = 53, a.LINK = 54, a.MAP = 55, a.MENU = 56, a.META = 57, a.NOFRAMES = 58, a.NOSCRIPT = 59, a.OBJECT = 60, a.OL = 61, a.OPTGROUP = 62, a.OPTION = 63, a.P = 64, a.PARAM = 65, a.PRE = 66, a.Q = 67, a.S = 68, a.SAMP = 69, a.SCRIPT = 70, a.SELECT = 71, a.SMALL = 72, a.SPAN = 73, a.STRIKE = 74, a.STRONG = 75, a.STYLE = 76, a.SUB = 77, a.SUP = 78, a.TABLE = 79, a.TBODY = 80, a.TD = 81, a.TEXTAREA = 82, a.TFOOT = 83, a.TH = 84, a.THEAD = 85, a.TITLE = 86, a.TR = 87, a.TT = 88, a.U = 89, a.UL = 90, a.VAR = 91, a.NOINDEX =
                        100, a
                }), lr = [17, 18, 38, 32, 39, 15, 11, 7, 1], pt = function () {
                    var a = "first(-|\\.|_|\\s){0,2}name last(-|\\.|_|\\s){0,2}name zip postal phone address passport (bank|credit)(-|\\.|_|\\s){0,2}card card(-|\\.|_|\\s){0,2}number card(-|\\.|_|\\s){0,2}holder cvv card(-|\\.|_|\\s){0,2}exp card(-|\\.|_|\\s){0,2}name card.*month card.*year card.*month card.*year password email birth(-|\\.|_|\\s){0,2}(day|date) second(-|\\.|_|\\s){0,2}name third(-|\\.|_|\\s){0,2}name patronymic middle(-|\\.|_|\\s){0,2}name birth(-|\\.|_|\\s){0,2}place house street city flat state".split(" ");
                    return {
                        xk: new RegExp("(" + J("|", a) + ")", "i"),
                        Hk: new RegExp("(" + J("|", a.concat("\u0438\u043c\u044f;\u0444\u0430\u043c\u0438\u043b\u0438\u044f;\u043e\u0442\u0447\u0435\u0441\u0442\u0432\u043e;\u0438\u043d\u0434\u0435\u043a\u0441;\u0442\u0435\u043b\u0435\u0444\u043e\u043d;\u0430\u0434\u0440\u0435\u0441;\u043f\u0430\u0441\u043f\u043e\u0440\u0442;\u041d\u043e\u043c\u0435\u0440(-|\\.|_|\\s){0,2}\u043a\u0430\u0440\u0442\u044b;\u0434\u0430\u0442\u0430(-|\\.|_|\\s){0,2} \u0440\u043e\u0436\u0434\u0435\u043d\u0438\u044f;\u0434\u043e\u043c;\u0443\u043b\u0438\u0446\u0430;\u043a\u0432\u0430\u0440\u0442\u0438\u0440\u0430;\u0433\u043e\u0440\u043e\u0434;\u043e\u0431\u043b\u0430\u0441\u0442\u044c".split(";"))) +
                            ")", "i"),
                        vk: /ym-record-keys/i,
                        $i: "\u2022",
                        Gk: 88
                    }
                }(), uk = hb(u(pt.$i, O)), kd = !0, Ng = "", Og = !1, Pg = !0, Qg = !1, qn = la(function (a, c) {
                    var b = E([a, "efv." + c.event], D);
                    c.N = A(t(O, b), c.N);
                    return c
                }), nl = x(function (a) {
                    var c = [], b = [], d = [];
                    a.document.attachEvent && !a.opera && (c.push(Ef), b.push(sr), b.push(tr));
                    a.document.addEventListener ? c.push(sk) : (b.push(rk), d.push(sk));
                    c = xa([{target: a, type: "window", event: "beforeunload", N: [B]}, {
                        target: a,
                        type: "window",
                        event: "unload",
                        N: [B]
                    }, {event: "click", N: [Te]}, {event: "dblclick", N: [Te]}, {
                        event: "mousedown",
                        N: [Te]
                    }, {event: "mouseup", N: [vr]}, {event: "keydown", N: [wr]}, {
                        event: "keypress",
                        N: [xr]
                    }, {event: "copy", N: [wk]}, {event: "blur", N: c}, {event: "focusin", N: b}, {
                        event: "focusout",
                        N: d
                    }], !a.document.attachEvent || a.opera ? [{
                        target: a,
                        type: "window",
                        event: "focus",
                        N: [Wh]
                    }, {
                        target: a,
                        type: "window",
                        event: "blur",
                        N: [Ef]
                    }] : [], a.document.addEventListener ? [{event: "focus", N: [rk]}, {
                        event: "change",
                        N: [tk]
                    }, {event: "submit", N: [yk]}] : [{type: "formInput", event: "change", N: [tk]}, {
                        type: "form",
                        event: "submit",
                        N: [yk]
                    }]);
                    return pn(a, c)
                }), nn =
                    x(function (a) {
                        return xa(yc(a) ? [{target: a, type: "document", event: "mouseleave", N: [yr]}] : [])
                    }), qt = ["submit", "beforeunload", "unload"], rt = x(function (a, c) {
                    var b = c(a);
                    return M(function (d, e) {
                        d[e.type + ":" + e.event] = e.N;
                        return d
                    }, {}, b)
                }), st = u(nl, function (a, c, b, d) {
                    return rt(c, a)[b + ":" + d] || []
                }), on = /^\s*function submit\(\)/, tt = C("fw.p", function (a, c) {
                    var b;
                    if (!(b = c.Dh || !c.Kb)) {
                        var d = H(a), e = !1;
                        b = d.o("hitParam", {});
                        var f = N(c);
                        b[f] && (d = d.o("counters", {}), e = !(!Yd(c.ba) || d[f]));
                        b[f] = 1;
                        b = e
                    }
                    if (b) return I.resolve(B);
                    b = new ot(a,
                        st);
                    return jn(a, c, b, nl, qt)
                }), ih, ol = (ih = function (a) {
                    function c(b, d, e, f) {
                        void 0 === f && (f = 0);
                        d = a.call(this, b, d, e) || this;
                        d.Se = 0;
                        d.Mb = 0;
                        d.Re = 0;
                        d.buffer = [];
                        d.rd = 2E3;
                        d.ca = id(b);
                        d.jd();
                        d.Re = f;
                        return d
                    }

                    Ka(c, a);
                    c.prototype.Af = function (b) {
                        return Z(Boolean, this.ca.O("ag", b))
                    };
                    c.prototype.zf = function (b, d) {
                        var e = this;
                        b(Sa(D(this.l, "wv2.b.st"), function (f) {
                            e.send(f, d)
                        }))
                    };
                    c.prototype.Bj = function (b, d) {
                        var e = this;
                        na(this.l, this.nf);
                        var f = Math.ceil(this.ra.$a(d) / 63E4), g = this.ra.kd(d, f);
                        z(function (h, k) {
                            var l, m = y({}, b,
                                (l = {}, l.data = h, l.partNum = k + 1, l.end = k + 1 === f, l));
                            l = e.ra.Ca([m], !1);
                            e.zf(l, [m])
                        }, g);
                        this.jd()
                    };
                    c.prototype.send = function (b, d) {
                        var e = this;
                        this.ca.O("se", d);
                        return a.prototype.send.call(this, b, d).then(O, function () {
                            e.ca.O("see", d)
                        })
                    };
                    c.Bf = function (b, d, e, f, g) {
                        c.ud["" + b + d] || (this.ud[d] = new c(g, f, e, "m" === d ? 31457280 : 0));
                        return this.ud[d]
                    };
                    c.prototype.Mi = function () {
                        return this.Re && this.Se >= this.Re
                    };
                    c.prototype.push = function (b) {
                        var d = this;
                        if (!this.Mi()) {
                            this.ca.O("p", b);
                            var e = this.ra.Ua(b), f = this.ra.$a(e);
                            7E5 <
                            f ? this.Bj(b, e) : (e = this.Af(this.buffer.concat([b])), e = M(function (g, h) {
                                return g + d.ra.$a(d.ra.Ua(h))
                            }, 0, e), this.Mb + e + f >= 7E5 * .7 && this.flush(), this.buffer.push(b), this.Mb += f)
                        }
                    };
                    c.prototype.D = function (b, d) {
                        this.ca.D([b], d)
                    };
                    c.prototype.oa = function (b, d) {
                        this.ca.oa([b], d)
                    };
                    c.prototype.flush = function () {
                        var b = this.buffer.concat(this.Af(this.buffer));
                        if (b.length) {
                            this.buffer = [];
                            this.Se += this.Mb;
                            this.Mb = 0;
                            var d = this.ra.Ca(b);
                            this.zf(d, b)
                        }
                    };
                    return c
                }(ll), ih.ud = {}, ih), $a = function () {
                    function a(c, b, d) {
                        this.Qi = "wv2.c";
                        this.Zb = [];
                        this.ma = [];
                        this.l = c;
                        this.K = Cf(c, this, d, this.Qi);
                        this.F = b;
                        this.lb = this.F.ei();
                        this.start = this.K.J(this.start, "st");
                        this.stop = this.K.J(this.stop, "sp")
                    }

                    a.prototype.start = function () {
                        var c = this;
                        this.Zb = A(function (b) {
                            var d = b[0], e = b[2];
                            b = F(c.K.J(b[1], d[0]), c);
                            return c.lb.D(e || c.l, d, b)
                        }, this.ma)
                    };
                    a.prototype.stop = function () {
                        z(ha, this.Zb)
                    };
                    a.prototype.$ = function (c) {
                        return this.F.va().$(c)
                    };
                    return a
                }(), gn = ["checkbox", "radio"], hn = /pwd|value|password/i, jh = X("location.href"), ut = function (a) {
                    function c(b,
                               d, e) {
                        d = a.call(this, b, d, e) || this;
                        d.ua = {elements: [], attributes: []};
                        d.index = 0;
                        d.oe = d.K.J(d.oe, "o");
                        d.Dd = d.K.J(d.Dd, "io");
                        d.sd = d.K.J(d.sd, "ao");
                        d.Ce = d.K.J(d.Ce, "a");
                        d.Ae = d.K.J(d.Ae, "at");
                        d.De = d.K.J(d.De, "r");
                        d.Be = d.K.J(d.Be, "c");
                        d.Aa = new b.MutationObserver(d.oe);
                        return d
                    }

                    Ka(c, a);
                    c.prototype.start = function () {
                        this.Aa.observe(this.l.document.documentElement, {
                            attributes: !0,
                            characterData: !0,
                            childList: !0,
                            subtree: !0,
                            attributeOldValue: !0,
                            characterDataOldValue: !0
                        })
                    };
                    c.prototype.stop = function () {
                        this.Aa.disconnect()
                    };
                    c.prototype.sd = function (b) {
                        var d = b.target;
                        b = b.attributeName;
                        var e = this.ua.elements.indexOf(d);
                        -1 === e && (e = this.ua.elements.push(d) - 1, this.ua.attributes[e] = {});
                        this.ua.attributes[e] || (this.ua.attributes[e] = {});
                        e = this.ua.attributes[e];
                        var f = d.getAttribute(b);
                        e[b] = ge(this.l, d, b, f, this.F.Tb()).value
                    };
                    c.prototype.Dd = function (b) {
                        function d(k) {
                            var l = Pb(e.l)(k, f);
                            return -1 === l ? (f.push(k), k = {Kd: {}}, g.push(k), k) : g[l]
                        }

                        var e = this, f = [], g = [];
                        z(function (k) {
                            var l = k.attributeName, m = k.removedNodes, p = k.oldValue, q = k.target,
                                r = k.nextSibling, v = k.previousSibling;
                            switch (k.type) {
                                case "attributes":
                                    e.sd(k);
                                    var w = d(q);
                                    w.Kd[l] || (w.Kd[l] = ge(e.l, q, l, p, e.F.Tb()).value);
                                    break;
                                case "childList":
                                    m && z(function (G) {
                                        w = d(G);
                                        w.xf || y(w, {xf: q, Th: r ? r : void 0, Uh: v ? v : void 0})
                                    }, ya(m));
                                    break;
                                case "characterData":
                                    w = d(q), w.yf || (w.yf = p)
                            }
                        }, b);
                        var h = this.F.va();
                        z(function (k, l) {
                            h.Le(k, g[l])
                        }, f)
                    };
                    c.prototype.oe = function (b) {
                        var d = this;
                        if (jh(this.l)) {
                            var e = this.F.L();
                            this.Dd(b);
                            z(function (f) {
                                var g = f.addedNodes, h = f.removedNodes, k = f.target;
                                switch (f.type) {
                                    case "childList":
                                        h &&
                                        h.length && d.De(ya(h), e);
                                        g && g.length && d.Ce(ya(g), e);
                                        break;
                                    case "characterData":
                                        d.Be(k, e)
                                }
                            }, b);
                            this.Ae(e)
                        } else this.stop()
                    };
                    c.prototype.Ae = function (b) {
                        var d = this;
                        z(function (e, f) {
                            var g = d.Ic();
                            d.F.X("mutation", {index: g, attributes: d.ua.attributes[f], target: d.$(e)}, "ac", b)
                        }, this.ua.elements);
                        this.ua.elements = [];
                        this.ua.attributes = []
                    };
                    c.prototype.Ce = function (b, d) {
                        var e = this, f = this.Ic();
                        this.F.va().Mc({
                            na: b, hd: function (g) {
                                g = A(function (h) {
                                    h = y({}, h);
                                    delete h.node;
                                    return h
                                }, g);
                                e.F.X("mutation", {index: f, na: g},
                                    "ad", d)
                            }
                        })
                    };
                    c.prototype.De = function (b, d) {
                        var e = this, f = this.Ic(), g = this.F.va(), h = A(function (k) {
                            var l = g.removeNode(k);
                            Mi(e.l, k, function (m) {
                                g.removeNode(m)
                            });
                            return l
                        }, b);
                        this.F.X("mutation", {index: f, na: h}, "re", d)
                    };
                    c.prototype.Be = function (b, d) {
                        var e = this.Ic();
                        this.F.X("mutation", {value: b.textContent, target: this.$(b), index: e}, "tc", d)
                    };
                    c.prototype.Ic = function () {
                        var b = this.index;
                        this.index += 1;
                        return b
                    };
                    return c
                }($a), vt = function () {
                    function a(c, b) {
                        var d = this;
                        this.Ec = [];
                        this.mb = [];
                        this.me = 1;
                        this.Wa = 0;
                        this.yb =
                            {};
                        this.Oc = {};
                        this.Vd = function (f) {
                            return d.mb.length ? K(f, d.mb) : !1
                        };
                        this.removeNode = function (f) {
                            var g = d.$(f), h = Ia(f);
                            if (h) return h = "NR:" + h.toLowerCase(), d.Vd(h) && d.ca.O(h, {data: {node: f, id: g}}), g
                        };
                        this.sb = function (f) {
                            var g = Ia(f);
                            if (g) {
                                var h = f.__ym_indexer;
                                h || (h = d.me, f.__ym_indexer = h, d.me += 1, g = "NA:" + g.toLowerCase(), d.Vd(g) && d.ca.O(g, {
                                    data: {
                                        node: f,
                                        id: h
                                    }
                                }));
                                return h
                            }
                            return null
                        };
                        this.l = c;
                        var e = Cf(c, this, "i");
                        this.ca = id(c);
                        this.options = b;
                        this.start = e.J(this.start, "st");
                        this.stop = e.J(this.stop, "sp");
                        this.$ =
                            e.J(this.$, "i");
                        this.Le = e.J(this.Le, "o");
                        this.Mc = e.J(this.Mc, "a");
                        this.removeNode = e.J(this.removeNode, "r");
                        this.ga = e.J(this.ga, "s")
                    }

                    a.prototype.Le = function (c, b) {
                        var d = this.sb(c);
                        Ua(d) || (this.Oc[d] && this.$(c), this.Oc[d] = b)
                    };
                    a.prototype.D = function (c, b, d) {
                        c = "" + b + c;
                        this.mb.push(c);
                        this.Vd(c) || this.mb.push(c);
                        this.ca.D([c], d)
                    };
                    a.prototype.oa = function (c, b, d) {
                        var e = "" + b + c;
                        this.mb = Z(function (f) {
                            return f !== e
                        }, this.mb);
                        this.ca.oa([e], d)
                    };
                    a.prototype.start = function () {
                        this.Wa = V(this.l, t(F(this.ga, this, !1), this.start),
                            50, "i.s")
                    };
                    a.prototype.stop = function () {
                        this.flush();
                        na(this.l, this.Wa);
                        this.Ec = []
                    };
                    a.prototype.Mc = function (c) {
                        var b = this, d = [], e = 0, f = {hd: c.hd, result: [], Nc: 0, na: d};
                        this.Ec.push(f);
                        z(function (g) {
                            Mi(b.l, g, function (h) {
                                var k = b.sb(h);
                                Ua(k) || (d.push(h), b.yb[k] && b.$(h), b.yb[k] = {node: h, event: f, Pj: e}, e += 1)
                            })
                        }, c.na)
                    };
                    a.prototype.$ = function (c) {
                        if (c === this.l) return 0;
                        var b = this.sb(c), d = this.yb[b], e = this.Xh(b), f = e.xf, g = e.Kd, h = e.yf, k = e.Th,
                            l = e.Uh;
                        if (d) {
                            e = d.event;
                            d = d.Pj;
                            var m = this.l.document.documentElement === c;
                            k =
                                k || c.nextSibling;
                            var p = l || c.previousSibling;
                            l = !m && k ? this.sb(k) : null;
                            p = !m && p ? this.sb(p) : null;
                            m = this.l;
                            k = this.options;
                            f = this.sb(f || c.parentNode || c.parentElement) || 0;
                            var q = g, r = void 0;
                            void 0 === p && (p = null);
                            void 0 === l && (l = null);
                            void 0 === q && (q = {});
                            void 0 === r && (r = Ia(c));
                            if (W(r)) c = void 0; else {
                                g = {
                                    id: b,
                                    ze: p !== f ? p : null,
                                    next: l !== f ? l : null,
                                    parent: f,
                                    name: r.toLowerCase(),
                                    node: c
                                };
                                if (If(c)) {
                                    if (h = dn(c, h), g.attributes = {}, g.content = h) if (c = jd(m, c)) g.content = "" !== h.trim() ? wn(m, h) : h, g.hidden = c
                                } else h = en(m, c, k, q, r), m = h.vb,
                                    g.attributes = h.ih, m && (g.hidden = m), c.namespaceURI && pc(c.namespaceURI, "svg") && (g.Vf = c.namespaceURI);
                                c = g
                            }
                            if (W(c)) return;
                            delete this.yb[b];
                            e.result[d] = c;
                            e.Nc += 1;
                            e.na.length === e.Nc && e.hd(e.result)
                        }
                        return b
                    };
                    a.prototype.flush = function () {
                        this.ga(!0)
                    };
                    a.prototype.ga = function (c) {
                        var b = this;
                        if (jh(this.l)) {
                            var d = da(this.yb);
                            d = A(function (e) {
                                return b.yb[e].node
                            }, d);
                            d = jc(d, this.$);
                            c = c ? Rj(B) : Pj(this.l, 20);
                            d(c);
                            this.Ec = Z(function (e) {
                                return e.Nc !== e.result.length
                            }, this.Ec)
                        }
                    };
                    a.prototype.Xh = function (c) {
                        if (Ua(c)) return {};
                        var b = this.Oc[c];
                        return b ? (delete this.Oc[c], b) : {}
                    };
                    return a
                }(), wt = ["input", "change", "keyup", "paste", "cut"], xt = function (a) {
                    function c(b, d, e) {
                        b = a.call(this, b, d, e) || this;
                        b.inputs = {};
                        b.Gd = !1;
                        b.Xc = b.K.J(b.Xc, "ii");
                        b.Yc = b.K.J(b.Yc, "ir");
                        b.cd = b.K.J(b.cd, "ri");
                        b.od = b.K.J(b.od, "ur");
                        b.Ud = b.K.J(b.Ud, "ce");
                        b.Lc = b.K.J(b.Lc, "vc");
                        return b
                    }

                    Ka(c, a);
                    c.prototype.start = function () {
                        var b = this, d = this.F.va();
                        this.Gd = this.rh();
                        z(function (e) {
                            e = e.toLowerCase();
                            d.D(e, "NA:", F(b.Xc, b));
                            d.D(e, "NR:", F(b.Yc, b))
                        }, Hf);
                        this.Zb = [this.lb.D(this.l.document,
                            wt, F(this.Ud, this)), function () {
                            z(function (e) {
                                e = e.toLowerCase();
                                d.oa(e, "NA:", b.Xc);
                                d.oa(e, "NR:", b.Yc)
                            }, Hf);
                            z(b.od, da(b.inputs))
                        }]
                    };
                    c.prototype.od = function (b) {
                        if (this.Gd) {
                            var d = this.inputs[b];
                            d && (b = d.jj, d = d.element, b && this.l.Object.defineProperty(d, this.Jc(d), b))
                        }
                    };
                    c.prototype.Yc = function (b) {
                        b && this.od(b.data.id)
                    };
                    c.prototype.Xc = function (b) {
                        b && (b = b.data, this.cd(b.node, b.id))
                    };
                    c.prototype.Jc = function (b) {
                        return Ie(b) ? "checked" : "value"
                    };
                    c.prototype.Ud = function (b) {
                        if (b = b.target) {
                            var d = this.Jc(b);
                            this.Lc(b[d],
                                b)
                        }
                    };
                    c.prototype.Lc = function (b, d) {
                        var e = this.$(d), f = this.inputs[e];
                        if (!f && (f = this.cd(f, e), !f)) return;
                        e = f.th;
                        var g = f.value, h = this.Jc(d), k = !b || K(typeof b, ["string", "boolean", "number"]),
                            l = this.F.Tb().ae;
                        k && b !== g && (g = ge(this.l, d, h, b, this.F.Tb()).value, e ? this.F.X("event", {
                            target: this.$(d),
                            checked: !!b
                        }, "change") : (e = Jc(this.l, d, l), l = e.pb, this.F.X("event", {
                            value: g,
                            hidden: e.wb && !l,
                            target: this.$(d)
                        }, "change")), f.value = b)
                    };
                    c.prototype.cd = function (b, d) {
                        var e = this;
                        if (!Af(b) || "__ym_input_override_test" === b.getAttribute("class") ||
                            this.inputs[d]) return null;
                        var f = Ie(b), g = this.Jc(b), h = {element: b, th: f, value: b[g]};
                        this.inputs[d] = h;
                        this.Gd && Kb(this.l, function () {
                            var k = e.l.Object.getOwnPropertyDescriptor(Object.getPrototypeOf(b), g) || {},
                                l = e.l.Object.getOwnPropertyDescriptor(b, g) || {}, m = y({}, k, l);
                            if (Aa("((set)?(\\s?" + g + ")?)?", m.set)) {
                                try {
                                    e.l.Object.defineProperty(b, g, y({}, m, {
                                        configurable: !0, set: function (p) {
                                            e.Lc(p, this);
                                            return m.set.call(this, p)
                                        }
                                    }))
                                } catch (p) {
                                }
                                h.jj = m
                            }
                        });
                        return h
                    };
                    c.prototype.rh = function () {
                        var b = !0, d = ab(this.l)("input");
                        try {
                            d = ab(this.l)("input");
                            d.value = "INPUT_VALUE";
                            d.style.setProperty("display", "none", "important");
                            d.setAttribute("type", "text");
                            d.setAttribute("class", "__ym_input_override_test");
                            var e = this.l.Object.getOwnPropertyDescriptor(Object.getPrototypeOf(d), "value") || {},
                                f = this.l.Object.getOwnPropertyDescriptor(d, "value") || {}, g = y({}, e, f);
                            this.l.Object.defineProperty(d, "value", y({}, g, {
                                configurable: !0, set: function (h) {
                                    return g.set.call(d, h)
                                }
                            }));
                            "INPUT_VALUE" !== d.value && (b = !1);
                            d.value = "INPUT_TEST";
                            "INPUT_TEST" !==
                            d.value && (b = !1)
                        } catch (h) {
                            b = !1
                        }
                        return b
                    };
                    return c
                }($a), yt = function (a) {
                    function c(b, d, e) {
                        b = a.call(this, b, d, e) || this;
                        b.fb = {width: 0, height: 0, Ab: 0, Bb: 0, orientation: 0};
                        b.ma.push([["resize"], b.hj]);
                        b.ma.push([["orientationchange"], b.fj]);
                        return b
                    }

                    Ka(c, a);
                    c.prototype.start = function () {
                        a.prototype.start.call(this);
                        this.ng()
                    };
                    c.prototype.hj = function () {
                        var b = this.Sd();
                        this.zi(b) && (this.fb = b, this.og(b))
                    };
                    c.prototype.fj = function () {
                        var b = this.Sd();
                        this.fb.orientation !== b.orientation && (this.fb = b, this.zj(b))
                    };
                    c.prototype.Rf =
                        function (b) {
                            return !b.height || !b.width || !b.Bb || !b.Ab
                        };
                    c.prototype.zi = function (b) {
                        return b.height !== this.fb.height || b.width !== this.fb.width
                    };
                    c.prototype.Sd = function () {
                        var b = this.F.rb(), d = Qc(this.l), e = d[0];
                        d = d[1];
                        b = b.Rd();
                        return {
                            width: e,
                            height: d,
                            Bb: b ? b.scrollWidth : 0,
                            Ab: b ? b.scrollHeight : 0,
                            orientation: this.F.rb().hi()
                        }
                    };
                    c.prototype.zj = function (b) {
                        var d;
                        void 0 === d && (d = this.F.L());
                        this.F.X("event", {
                            width: b.width,
                            height: b.height,
                            orientation: b.orientation
                        }, "deviceRotation", d)
                    };
                    c.prototype.og = function (b, d) {
                        void 0 ===
                        d && (d = this.F.L());
                        this.F.X("event", {width: b.width, height: b.height, Bb: b.Bb, Ab: b.Ab}, "resize", d)
                    };
                    c.prototype.ng = function () {
                        var b = this.Sd();
                        this.Rf(b) ? V(this.l, F(this.ng, this), 300) : (this.Rf(this.fb) && (this.fb = b), this.og(b, 0))
                    };
                    return c
                }($a), bf = function () {
                    function a(c) {
                        this.index = 0;
                        this.Hb = {};
                        this.l = c
                    }

                    a.prototype.zc = function (c, b, d) {
                        void 0 === d && (d = {});
                        var e = fa(this.l), f = this.index;
                        this.index += 1;
                        this.Hb[f] = {Wa: 0, dc: !1, Rh: c, Nb: [], fe: e(aa)};
                        var g = this;
                        return function () {
                            var h = Ca(arguments), k = d.ob && !g.Hb[f].dc,
                                l = g.Hb[f];
                            na(g.l, l.Wa);
                            l.Nb = h;
                            l.dc = !0;
                            var m = e(aa);
                            if (k || m - l.fe >= b) c.apply(void 0, h), l.fe = m;
                            l.Wa = V(g.l, function () {
                                k || (c.apply(void 0, h), l.fe = e(aa));
                                l.dc = !1;
                                l.Nb = []
                            }, b, "th")
                        }
                    };
                    a.prototype.flush = function () {
                        var c = this;
                        z(function (b) {
                            var d = c.Hb[b], e = d.Wa, f = d.Rh, g = d.Nb;
                            d.dc && (c.Hb[b].dc = !1, f.apply(void 0, g), na(c.l, e))
                        }, da(this.Hb))
                    };
                    return a
                }(), zt = function (a) {
                    function c(b, d, e) {
                        d = a.call(this, b, d, e) || this;
                        d.Dg = [];
                        d.$e = {x: 0, y: 0};
                        d.Da = new bf(b);
                        d.ad = d.K.J(d.ad, "o");
                        d.ma.push([["scroll"], d.ij]);
                        return d
                    }

                    Ka(c,
                        a);
                    c.prototype.start = function () {
                        a.prototype.start.call(this);
                        this.F.X("event", {
                            x: Math.max(this.l.scrollX, 0),
                            y: Math.max(this.l.scrollY, 0),
                            page: !0,
                            target: -1
                        }, "scroll", 0)
                    };
                    c.prototype.stop = function () {
                        a.prototype.stop.call(this);
                        this.Da.flush()
                    };
                    c.prototype.ij = function (b) {
                        if (this.F.rb().Mf()) this.ad(b); else {
                            var d = b.target, e = Z(function (f) {
                                return f[0] === d
                            }, this.Dg).pop();
                            e ? e = e[1] : (e = this.Da.zc(F(this.ad, this), 100, {ob: !0}), this.Dg.push([d, e]));
                            e(b)
                        }
                    };
                    c.prototype.ad = function (b) {
                        var d = this.F.rb().Rd();
                        b = b.target;
                        var e = this.Ub(b);
                        d = Ja(ka(b), [d, this.l, this.l.document]);
                        var f = Math.max(e.left, 0);
                        e = Math.max(e.top, 0);
                        if (d) {
                            if (this.$e.x === f && this.$e.y === e) return;
                            this.$e = {x: f, y: e}
                        }
                        this.F.X("event", {x: f, y: e, page: d, target: d ? -1 : this.$(b)}, "scroll")
                    };
                    c.prototype.Ub = function (b) {
                        var d = {left: 0, top: 0};
                        if (!b) return d;
                        if (b.window === b) return {top: b.scrollY || 0, left: b.scrollX || 0};
                        var e = b.ownerDocument || b, f = b.documentElement, g = e.defaultView || e.parentWindow,
                            h = e.body;
                        return b !== e || (b = this.F.rb().Rd(), b) ? K(b, [f, h]) ? {
                            top: b.scrollTop ||
                                g.scrollY, left: b.scrollLeft || g.scrollX
                        } : {top: b.scrollTop || 0, left: b.scrollLeft || 0} : d
                    };
                    return c
                }($a), At = ["mousemove", "mousedown", "mouseup", "click"], Bt = function (a) {
                    function c(b, d, e) {
                        d = a.call(this, b, d, e) || this;
                        d.ma.push([At, d.ej]);
                        d.Da = new bf(b);
                        d.Vc = d.K.J(d.Vc, "n");
                        d.Hj = d.K.J(d.Da.zc(F(d.Vc, d), 100), "t");
                        return d
                    }

                    Ka(c, a);
                    c.prototype.stop = function () {
                        a.prototype.stop.call(this);
                        this.Da.flush()
                    };
                    c.prototype.ej = function (b) {
                        var d = null;
                        try {
                            d = b.type
                        } catch (e) {
                            return
                        }
                        "mousemove" === d ? this.Hj(b) : this.Vc(b)
                    };
                    c.prototype.Vc =
                        function (b) {
                            var d = b.type, e = b.clientX;
                            e = void 0 === e ? null : e;
                            var f = b.clientY;
                            f = void 0 === f ? null : f;
                            b = b.target || this.l.document.elementFromPoint(e, f);
                            this.F.X("event", {x: e || 0, y: f || 0, target: this.$(b)}, d)
                        };
                    return c
                }($a), Ct = ["focus", "blur"], Dt = function (a) {
                    function c(b, d, e) {
                        b = a.call(this, b, d, e) || this;
                        b.ma.push([Ct, b.Sh]);
                        return b
                    }

                    Ka(c, a);
                    c.prototype.Sh = function (b) {
                        var d = b.target;
                        b = b.type;
                        this.F.X("event", {target: this.$(d === this.l ? this.l.document.documentElement : d)}, b)
                    };
                    return c
                }($a), Et = x(function (a) {
                    var c = Ma(a.getSelection,
                        "getSelection");
                    return c ? F(c, a) : B
                }), Ft = t(Et, ha), Gt = ["mousemove", "touchmove", "mousedown", "touchdown", "select"],
                Ht = /text|search|password|tel|url/, It = function (a) {
                    function c(b, d, e) {
                        b = a.call(this, b, d, e) || this;
                        b.Wd = !1;
                        b.ma.push([Gt, b.wi]);
                        return b
                    }

                    Ka(c, a);
                    c.prototype.wi = function (b) {
                        var d = this.F, e = b.type, f = b.which;
                        b = b.target;
                        if ("mousemove" !== e || 1 === f) (e = "select" === e ? this.li(b) : this.ji()) && e.start !== e.end ? (this.Wd = !0, d.X("event", e, "selection")) : this.Wd && (this.Wd = !1, d.X("event", {
                            start: 0,
                            end: 0
                        }, "selection"))
                    };
                    c.prototype.ji = function () {
                        var b = Ft(this.l);
                        if (b && 0 < b.rangeCount) {
                            b = b.getRangeAt(0) || this.l.document.createRange();
                            var d = this.$(b.startContainer), e = this.$(b.endContainer);
                            if (!W(d) && !W(e)) return {start: b.startOffset, end: b.endOffset, yg: d, pf: e}
                        }
                    };
                    c.prototype.li = function (b) {
                        if (Ht.test(b.type || "")) {
                            var d = this.$(b);
                            if (!W(d)) return {start: b.selectionStart, end: b.selectionEnd, target: d}
                        }
                    };
                    return c
                }($a), Jt = {focus: "windowfocus", blur: "windowblur"}, Kt = function (a) {
                    function c(b, d, e) {
                        b = a.call(this, b, d, e) || this;
                        b.visibility =
                            null;
                        W(b.l.document.hidden) ? W(b.l.document.msHidden) ? W(b.l.document.webkitHidden) || (b.visibility = {
                            hidden: "webkitHidden",
                            event: "webkitvisibilitychange"
                        }) : b.visibility = {
                            hidden: "msHidden",
                            event: "msvisibilitychange"
                        } : b.visibility = {hidden: "hidden", event: "visibilitychange"};
                        b.handleEvent = b.K.J(b.handleEvent, "e");
                        return b
                    }

                    Ka(c, a);
                    c.prototype.start = function () {
                        this.Zb = [this.lb.D(this.l, this.visibility ? [this.visibility.event] : ["focus", "blur"], F(this.handleEvent, this))]
                    };
                    c.prototype.handleEvent = function (b) {
                        this.F.X("event",
                            {}, Jt[this.visibility ? this.l.document[this.visibility.hidden] ? "blur" : "focus" : b.type])
                    };
                    return c
                }($a), Lt = ["touchmove", "touchstart", "touchend", "touchcancel", "touchforcechange"], Mt = function (a) {
                    function c(b, d, e) {
                        d = a.call(this, b, d, e) || this;
                        d.md = {};
                        d.scrolling = !1;
                        d.lg = 0;
                        d.ma.push([["scroll"], d.xj, d.l.document]);
                        d.ma.push([Lt, d.Lj, d.l.document]);
                        d.Da = new bf(b);
                        d.Yb = d.K.J(d.Yb, "nh");
                        d.Ij = d.K.J(d.Da.zc(d.Yb, d.F.rb().Mf() ? 0 : 50, {ob: !0}), "th");
                        return d
                    }

                    Ka(c, a);
                    c.prototype.xj = function () {
                        var b = this;
                        this.scrolling =
                            !0;
                        na(this.l, this.lg);
                        this.lg = V(this.l, function () {
                            b.scrolling = !1
                        }, 150)
                    };
                    c.prototype.Lj = function (b) {
                        var d = this, e = "touchcancel" === b.type || "touchend" === b.type;
                        b.changedTouches && 0 < b.changedTouches.length && z(function (f) {
                            var g = d.oi(f);
                            f.__ym_touch_id = g;
                            e && delete d.md[f.identifier]
                        }, ya(b.changedTouches));
                        "touchmove" === b.type ? this.scrolling ? this.Yb(b) : this.Ij(b, this.F.L()) : this.Yb(b)
                    };
                    c.prototype.oi = function (b) {
                        this.md[b.identifier] || (this.md[b.identifier] = Ph());
                        return this.md[b.identifier]
                    };
                    c.prototype.Yb =
                        function (b, d) {
                            void 0 === d && (d = this.F.L());
                            var e = b.type, f = A(function (g) {
                                return {
                                    id: g.__ym_touch_id,
                                    x: Math.round(g.clientX),
                                    y: Math.round(g.clientY),
                                    force: g.force
                                }
                            }, ya(b.changedTouches));
                            0 < f.length && this.F.X("event", {touches: f, target: this.$(b.target)}, e, d)
                        };
                    return c
                }($a), Nt = function (a) {
                    function c(b, d, e) {
                        b = a.call(this, b, d, e) || this;
                        b.ma.push([["load"], b.dj, b.l.document]);
                        return b
                    }

                    Ka(c, a);
                    c.prototype.dj = function (b) {
                        b = b.target;
                        "IMG" === Ia(b) && b.getAttribute("srcset") && this.F.X("event", {
                                target: this.$(b),
                                value: b.currentSrc
                            },
                            "srcset")
                    };
                    return c
                }($a), Ot = function (a) {
                    function c(b, d, e) {
                        d = a.call(this, b, d, e) || this;
                        d.Rg = 1;
                        d.Da = new bf(b);
                        d.rc = d.K.J(d.rc, "z");
                        return d
                    }

                    Ka(c, a);
                    c.prototype.start = function () {
                        if (this.If()) {
                            a.prototype.start.call(this);
                            this.rc();
                            var b = this.lb.D(n(this.l, "visualViewport"), ["resize"], this.Da.zc(this.rc, 10));
                            this.Zb.push(b)
                        }
                    };
                    c.prototype.stop = function () {
                        a.prototype.stop.call(this);
                        this.Da.flush()
                    };
                    c.prototype.rc = function () {
                        var b = this.If();
                        b && b !== this.Rg && (this.Rg = b, this.Aj(b))
                    };
                    c.prototype.If = function () {
                        var b =
                            Je(this.l);
                        return b ? b[2] : null
                    };
                    c.prototype.Aj = function (b) {
                        var d = Zf(this.l);
                        this.F.X("event", {x: d.x, y: d.y, level: b}, "zoom")
                    };
                    return c
                }($a), Wd, cf = {
                    91: "super",
                    93: "super",
                    224: "super",
                    18: "alt",
                    17: "ctrl",
                    16: "shift",
                    9: "tab",
                    8: "backspace",
                    46: "delete"
                }, pl = {"super": 1, pk: 2, alt: 3, shift: 4, Qk: 5, "delete": 6, lk: 6},
                Pt = [4, 9, 8, 32, 37, 38, 39, 40, 46], ql = (Wd = {}, Wd["1"] = {
                    91: "&#8984;",
                    93: "&#8984;",
                    224: "&#8984;",
                    18: "&#8997;",
                    17: "&#8963;",
                    16: "&#8679;",
                    9: "&#8677;",
                    8: "&#9003;",
                    46: "&#9003;"
                }, Wd["2"] = {
                    91: "&#xff;", 93: "&#xff;", 224: "&#xff;",
                    18: "Alt", 17: "Ctrl", 16: "Shift", 9: "Tab", 8: "Backspace", 46: "Delete"
                }, Wd.Ui = {32: "SPACEBAR", 37: "&larr;", 38: "&uarr;", 39: "&rarr;", 40: "&darr;", 13: "Enter"}, Wd),
                Qt = /flash/, Rt = /ym-disable-keys/, St = /^&#/, Tt = function (a) {
                    function c(b, d, e) {
                        d = a.call(this, b, d, e) || this;
                        d.tb = {};
                        d.Sa = 0;
                        d.Ia = [];
                        d.Bg = [];
                        d.Cc = 0;
                        d.gg = 0;
                        d.ma.push([["keydown"], d.ti]);
                        d.ma.push([["keyup"], d.ui]);
                        d.ah = -1 !== (n(b, "navigator.appVersion") || "").indexOf("Mac") ? "1" : "2";
                        d.Rc = d.K.J(d.Rc, "v");
                        d.Id = d.K.J(d.Id, "ec");
                        d.gd = d.K.J(d.gd, "sk");
                        d.Pd = d.K.J(d.Pd,
                            "gk");
                        d.Me = d.K.J(d.Me, "sc");
                        d.qc = d.K.J(d.qc, "cc");
                        d.reset = d.K.J(d.reset, "r");
                        d.ed = d.K.J(d.ed, "rs");
                        return d
                    }

                    Ka(c, a);
                    c.prototype.ti = function (b) {
                        if (this.Rc(b) && !this.Li(b)) {
                            var d = b.keyCode;
                            b.repeat || this.tb[d] || (this.tb[b.keyCode] = !0, cf[b.keyCode] && !this.Sa ? (this.Sa += 1, this.Me(b), this.reset(300)) : this.Sa ? (this.nh(), this.Fe(b), this.Id()) : (this.reset(), this.Fe(b)))
                        }
                    };
                    c.prototype.ui = function (b) {
                        if (this.Rc(b)) {
                            var d = b.keyCode, e = cf[b.keyCode];
                            this.tb[b.keyCode] && (this.tb[d] = !1);
                            e && this.Sa && (this.Sa = 0, this.tb =
                                {});
                            1 === this.Ia.length && (b = this.Ia[0], K(b.keyCode, Pt) && (this.gd([b], !0), this.reset()));
                            this.Ia = Z(t(ka(d), Tb), this.Ia);
                            na(this.l, this.Cc)
                        }
                    };
                    c.prototype.Rc = function (b) {
                        var d = this.l.document.activeElement;
                        b = b.target;
                        return !Ja(Boolean, [d && "OBJECT" === d.nodeName && Qt.test(d.getAttribute("type") || ""), "INPUT" === b.nodeName && "password" === b.getAttribute("type") && Rt.test(b.className)])
                    };
                    c.prototype.Id = function () {
                        this.Bg = this.Ia.slice(0);
                        na(this.l, this.Cc);
                        this.Cc = V(this.l, u(this.Bg, F(this.gd, this)), 0, "e.c")
                    };
                    c.prototype.gd = function (b, d) {
                        void 0 === d && (d = !1);
                        if (1 < b.length || d) {
                            var e = this.Pd(b);
                            this.F.X("event", {Sc: e}, "keystroke")
                        }
                    };
                    c.prototype.Pd = function (b) {
                        var d = this;
                        b = A(function (e) {
                            e = e.keyCode;
                            var f = cf[e], g = d.gi(e);
                            return {id: e, key: g, Of: !!f && St.test(g), Tc: f}
                        }, b);
                        return Eg(function (e, f) {
                            return (pl[e.Tc] || 100) - (pl[f.Tc] || 100)
                        }, b)
                    };
                    c.prototype.gi = function (b) {
                        return ql[this.ah][b] || ql.Ui[b] || String.fromCharCode(b)
                    };
                    c.prototype.Fe = function (b) {
                        K(b, this.Ia) || this.Ia.push(b)
                    };
                    c.prototype.Me = function (b) {
                        this.Fe(b);
                        this.qc()
                    };
                    c.prototype.qc = function () {
                        this.Sa ? V(this.l, this.qc, 100) : this.Ia = []
                    };
                    c.prototype.nh = function () {
                        na(this.l, this.gg)
                    };
                    c.prototype.reset = function (b) {
                        b ? this.gg = V(this.l, F(this.ed, this), b) : this.ed()
                    };
                    c.prototype.ed = function () {
                        this.Sa = 0;
                        this.Ia = [];
                        this.tb = {};
                        na(this.l, this.Cc)
                    };
                    c.prototype.Li = function (b) {
                        return b.target && "INPUT" === b.target.nodeName ? b.shiftKey || 32 === b.keyCode || "shift" === cf[b.keyCode] : !1
                    };
                    return c
                }($a), cn = ["sr", "sd", "\u043d"], Ut = /allow-same-origin/, Vt = function (a) {
                    function c(b, d,
                               e) {
                        d = a.call(this, b, d, e) || this;
                        d.ic = [];
                        d.Ld = {};
                        d.pe = d.K.J(d.pe, "fi");
                        d.qe = d.K.J(d.qe, "sd");
                        d.se = d.K.J(d.se, "src");
                        d.Aa = new b.MutationObserver(d.se);
                        return d
                    }

                    Ka(c, a);
                    c.prototype.start = function () {
                        a.prototype.start.call(this);
                        this.F.Tb().sc && this.F.va().D("iframe", "NA:", F(this.pe, this));
                        this.F.Ef().Nd().D(["sdr"], F(this.qe, this))
                    };
                    c.prototype.stop = function () {
                        a.prototype.stop.call(this);
                        z(function (b) {
                            b.F.stop()
                        }, this.ic)
                    };
                    c.prototype.se = function (b) {
                        var d = b.pop().target;
                        if (b = bb(function (f) {
                            return f.Lf ===
                                d
                        }, this.ic)) {
                            this.ic = Z(function (f) {
                                return f.Lf !== d
                            }, this.ic);
                            var e = b.F.Od();
                            try {
                                b.F.stop()
                            } catch (f) {
                            }
                            this.yc(d, e)
                        }
                    };
                    c.prototype.pe = function (b) {
                        if (b) {
                            var d = b.data.node;
                            this.Aa.observe(d, {attributes: !0, attributeFilter: ["src"]});
                            this.yc(d, b.data.id)
                        }
                    };
                    c.prototype.yc = function (b, d) {
                        var e = this;
                        this.Hi(b) && cc(this.l, b)(Sa(B, function () {
                            var f = e.F.yc(b.contentWindow, d);
                            e.ic.push({F: f, Lf: b})
                        }))
                    };
                    c.prototype.qe = function (b) {
                        var d = this, e = b.M;
                        b = b.data;
                        this.Ld[e] || (this.Ld[e] = {data: []});
                        var f = this.Ld[e];
                        f.data = f.data.concat(b);
                        this.l.isNaN(f.Fd) && z(function (g) {
                            "page" === g.type && (f.Fd = g.data.Ka - d.F.Ff())
                        }, f.data);
                        this.l.isNaN(f.Fd) || (this.F.ga(A(function (g) {
                            g.L += f.Fd;
                            g.L = d.l.Math.max(0, g.L);
                            return g
                        }, f.data)), f.data = [])
                    };
                    c.prototype.Hi = function (b) {
                        var d = b.getAttribute("src"), e = b.getAttribute("sandbox");
                        return b.getAttribute("_ym_ignore") || e && !e.match(Ut) || d && "about:blank" !== d && (d = Hc(this.l, d).host) && S(this.l).host !== d ? !1 : n(b, "contentWindow.location.href")
                    };
                    return c
                }($a), Wt = function (a) {
                    function c(b, d, e) {
                        b = a.call(this, b, d,
                            e) || this;
                        b.Ie = b.K.J(b.Ie, "ps");
                        return b
                    }

                    Ka(c, a);
                    c.prototype.start = function () {
                        this.F.va().Mc({na: [this.l.document.documentElement], hd: this.Ie})
                    };
                    c.prototype.Ie = function (b) {
                        var d = this.F.ii(), e = d.Yh(), f = S(this.l), g = f.host, h = f.protocol;
                        f = f.pathname;
                        var k = Qc(this.l), l = k[0];
                        k = k[1];
                        this.F.X("page", {
                            content: A(function (m) {
                                m = y({}, m);
                                delete m.node;
                                return m
                            }, b),
                            hf: e || "",
                            Jf: !!e,
                            viewport: {width: l, height: k},
                            title: this.l.document.title,
                            doctype: d.$h() || "",
                            ef: this.l.location.href,
                            Jg: this.l.navigator.userAgent,
                            referrer: this.l.document.referrer,
                            screen: {width: this.l.screen.width, height: this.l.screen.height},
                            location: {host: g, protocol: h, path: f},
                            Ka: this.F.Ff(),
                            ld: d.mi()
                        }, "page", 0)
                    };
                    return c
                }($a), Xt = ["addRule", "removeRule", "insertRule", "deleteRule"], kh = [[function (a) {
                    function c(b, d, e) {
                        b = a.call(this, b, d, e) || this;
                        b.hb = {};
                        b.jc = {};
                        b.ff = 0;
                        b.Zc = b.K.J(b.Zc, "a");
                        b.Gb = b.K.J(b.Gb, "sr");
                        b.$c = b.K.J(b.$c, "r");
                        b.ga = b.K.J(b.ga, "d");
                        return b
                    }

                    Ka(c, a);
                    c.prototype.start = function () {
                        var b = this.F.va();
                        b.D("style", "NA:", this.Zc);
                        b.D("style", "NR:", this.$c);
                        this.ga()
                    };
                    c.prototype.stop = function () {
                        var b = this;
                        a.prototype.stop.call(this);
                        var d = this.F.va();
                        d.oa("style", "NA:", this.Zc);
                        d.oa("style", "NR:", this.$c);
                        this.ga();
                        na(this.l, this.ff);
                        z(function (e) {
                            b.hb[e].sheet && b.jg(b.hb[e].sheet)
                        }, da(this.hb));
                        this.hb = {}
                    };
                    c.prototype.ga = function () {
                        var b = this;
                        z(function (d) {
                            var e = d[0];
                            d = d[1];
                            if (d.length) {
                                for (var f = [], g = d[0].L, h = [], k = 0; k < d.length; k += 1) {
                                    var l = d[k], m = l.L;
                                    delete l.L;
                                    m <= g + 50 ? f.push(l) : (h.push(f), g = m, f = [l])
                                }
                                f.length && h.push(f);
                                h.length && z(function (p) {
                                    b.F.X("event",
                                        {target: Ga(e), Oa: p}, "stylechange", g)
                                }, h);
                                delete b.jc[e]
                            }
                        }, pa(this.jc));
                        this.ff = V(this.l, this.ga, 100)
                    };
                    c.prototype.Gb = function (b, d, e, f, g) {
                        void 0 === f && (f = "");
                        void 0 === g && (g = -1);
                        this.jc[b] || (this.jc[b] = []);
                        this.jc[b].push({te: d, style: f, index: g, L: e})
                    };
                    c.prototype.kj = function (b, d) {
                        var e = this, f = b.addRule, g = b.removeRule, h = b.insertRule, k = b.deleteRule;
                        T(f) && (b.addRule = function (l, m, p) {
                            e.Gb(d, "a", e.F.L(), l + "{" + m + "}", p);
                            return f.call(b, l, m, p)
                        });
                        T(g) && (b.removeRule = function (l) {
                            e.Gb(d, "r", e.F.L(), "", l);
                            return g.call(b,
                                l)
                        });
                        T(h) && (b.insertRule = function (l, m) {
                            e.Gb(d, "a", e.F.L(), l, m);
                            return h.call(b, l, m)
                        });
                        T(k) && (b.deleteRule = function (l) {
                            e.Gb(d, "r", e.F.L(), "", l);
                            return k.call(b, l)
                        })
                    };
                    c.prototype.jg = function (b) {
                        var d = this;
                        z(function (e) {
                            var f = d.l.CSSStyleSheet.prototype[e];
                            T(f) && (b[e] = F(f, b))
                        }, Xt)
                    };
                    c.prototype.Kh = function (b) {
                        try {
                            return b.cssRules || b.rules
                        } catch (d) {
                            return null
                        }
                    };
                    c.prototype.Zc = function (b) {
                        var d = b.data;
                        b = d.id;
                        d = d.node;
                        if (d.sheet && !d.getAttribute("src") && !d.innerText) {
                            var e = d.sheet, f = this.Kh(e);
                            if (f && f.length) {
                                for (var g =
                                    [], h = 0; h < f.length; h += 1) g.push({style: f[h].cssText, index: h, te: "a"});
                                this.F.X("event", {Oa: g, target: b}, "stylechange")
                            }
                            this.kj(e, b);
                            this.hb[b] = d
                        }
                    };
                    c.prototype.$c = function (b) {
                        b = b.data.id;
                        var d = this.hb[b];
                        d && (delete this.hb[b], d.sheet && this.jg(d.sheet))
                    };
                    return c
                }($a), "ss"], [xt, "in"], [ut, "mu"], [yt, "r"], [zt, "sc"], [Bt, "mo"], [Dt, "f"], [It, "se"], [Kt, "wf"], [Mt, "t"], [Nt, "src"], [Ot, "z"], [Tt, "ks"]];
            kh.push([Vt, "if"]);
            kh.push([Wt, "pa"]);
            var Yt = x(function (a) {
                    var c = a.document;
                    return {
                        Rd: function () {
                            if (c.scrollingElement) return c.scrollingElement;
                            var b = 0 === c.compatMode.indexOf("CSS1") ? c.documentElement : c.body;
                            return n(c, "documentElement.scrollHeight") >= n(c, "body.scrollHeight") ? b : null
                        }, hi: function () {
                            var b = a.screen;
                            if (!b) return 0;
                            var d = bb(u(b, n), ["orientation", "mozOrientation", "msOrientation"]);
                            return n(b, d + ".angle") || 0
                        }, Dk: u(a, eb), Mf: u(a, od), Ck: u(a, Ne)
                    }
                }), Zt = function () {
                    function a(c, b) {
                        var d = this;
                        this.Wb = 0;
                        this.Ed = [];
                        this.Vb = null;
                        this.xa = this.lc = this.zg = !1;
                        this.Ka = 0;
                        this.ii = function () {
                            return d.page
                        };
                        this.Od = function () {
                            return d.Wb
                        };
                        this.Ff = function () {
                            return d.Ka
                        };
                        this.ei = function () {
                            return d.lb
                        };
                        this.Ef = function () {
                            return d.Vb
                        };
                        this.va = function () {
                            return d.Xd
                        };
                        this.L = function () {
                            return d.Qe ? d.l.Math.max(d.Qe(aa) - d.Ka, 0) : 0
                        };
                        this.Tb = function () {
                            return d.options
                        };
                        this.rb = function () {
                            return d.kh
                        };
                        this.X = function (f, g, h, k) {
                            void 0 === k && (k = d.L());
                            f = d.bi(f, g, h, k);
                            d.ga(f)
                        };
                        this.bi = function (f, g, h, k) {
                            void 0 === k && (k = d.L());
                            return {type: f, data: g, L: k, M: d.Wb, event: h}
                        };
                        this.ga = function (f) {
                            f = ca(f) ? f : [f];
                            d.zg && !d.lc ? d.xa ? (f = A(function (g) {
                                    return g.M ? g : y(g, {M: d.Wb})
                                }, f), d.Ef().pg(f)) :
                                (z(function (g) {
                                    ib(d.l, {name: "webvisorData", data: g})
                                }, f), f = A(Zm, f), d.ec(f)) : d.Ed = d.Ed.concat(f)
                        };
                        this.l = c;
                        var e = Cf(c, this, "R");
                        this.Ne = e.J(this.Ne, "s");
                        this.ga = e.J(this.ga, "sd");
                        e = H(c);
                        e.o("wv2e") && ff();
                        e.C("wv2e", !0);
                        this.options = b;
                        this.lb = ia(c);
                        this.Xd = new vt(this.l, b);
                        this.kh = Yt(c);
                        this.jf = A(function (f) {
                            return new f[0](c, d, f[1])
                        }, kh);
                        this.Fi();
                        this.page = an(this.l);
                        this.Ne()
                    }

                    a.prototype.start = function (c) {
                        this.zg = !0;
                        this.ec = c;
                        this.mg()
                    };
                    a.prototype.stop = function () {
                        jh(this.l) && (z(function (c) {
                                return c.stop()
                            },
                            this.jf), this.Xd.stop(), this.Vb && this.Vb.stop(), this.xa || this.X("event", {}, "eof"))
                    };
                    a.prototype.yc = function (c, b) {
                        var d = new a(c, y({}, this.options, {M: b}));
                        d.start(B);
                        return d
                    };
                    a.prototype.Fi = function () {
                        var c = this;
                        this.xa = !!this.options.M;
                        this.Wb = this.options.M || 0;
                        this.lc = !this.xa;
                        var b = this.options.Ig || [];
                        b.push(S(this.l).host);
                        this.Vb = bn(this.l, this, b);
                        b = this.Vb.Nd();
                        eb(this.l) ? this.lc && b.D(["i"], function (d) {
                            c.xa = d.xa;
                            c.lc = !1;
                            d.M && (c.Wb = d.M);
                            c.mg()
                        }) : this.lc = this.xa = !1
                    };
                    a.prototype.mg = function () {
                        var c =
                            Ed(this.Ed);
                        this.ga(c)
                    };
                    a.prototype.Ne = function () {
                        this.Qe = fg(this.l);
                        this.Ka = this.Qe(aa);
                        z(function (c) {
                            c.start()
                        }, this.jf);
                        this.Xd.start()
                    };
                    return a
                }(), fd = t(da, xc), ea,
                $t = (ea = {}, ea.mousemove = 0, ea.mouseup = 1, ea.mousedown = 2, ea.click = 3, ea.scroll = 4, ea.windowblur = 5, ea.windowfocus = 6, ea.focus = 7, ea.blur = 8, ea.eof = 9, ea.selection = 10, ea.change = 11, ea.input = 12, ea.touchmove = 13, ea.touchstart = 14, ea.touchend = 15, ea.touchcancel = 16, ea.touchforcechange = 17, ea.canvasMethod = 18, ea.canvasProperty = 19, ea.zoom = 20, ea.resize = 21, ea.mediaMethod =
                    22, ea.mediaProperty = 23, ea.keystroke = 24, ea.deviceRotation = 25, ea.fatalError = 26, ea.hashchange = 27, ea.stylechange = 28, ea),
                lh = la(function (a, c) {
                    var b;
                    return b = {}, b[fd(a)] = c, b
                }), au = function () {
                    function a(c) {
                        var b;
                        this.isSync = !1;
                        this.Qb = [];
                        this.sf = [];
                        this.l = c;
                        this.Xb = (b = {}, b.event = F(this.Hh, this), b.page = lh({page: 1}), b.mutation = lh({Vi: 1}), b.activity = lh({eh: 1}), b);
                        this.sf = [[["scroll"], {wj: 1}], [["selection"], {yj: 1}], [["change", "input"], {ph: 1}], [["keystroke"], {Pi: 1}, {Sc: 1}], [["zoom"], {ck: 1}], [["resize"], {sj: 1}],
                            [["windowfocus", "windowblur", "focus", "blur", "eof"], {$j: 1}], [["mousemove", "mouseup", "mousedown", "click"], {Ti: 1}], [["deviceRotation"], {Ch: 1}], [["fatalError"], {Mh: 1}], [["touchmove", "touchstart", "touchend", "touchcancel", "touchforcechange"], {Kj: 1}, {touches: 1}, {touches: 1}], [["hashchange"], {Ai: 1}], [["stylechange"], {Cj: 1}, {Oa: 1}, {Oa: 1}]]
                    }

                    a.prototype.Hh = function (c) {
                        var b, d, e = c.type, f = bb(t(xc, u(e, K)), this.sf);
                        f || Xa(ic("vem." + e));
                        "eof" === e && (this.isSync = !0);
                        var g = f[1], h = f[2];
                        f = f[3];
                        var k = c.aa;
                        return {
                            event: (b =
                                {
                                    type: $t[e],
                                    target: c.target,
                                    M: c.M
                                }, b[fd(g)] = h ? (d = {}, d[fd(h)] = f ? k[fd(f)] : k, d) : k, b)
                        }
                    };
                    a.prototype.Ca = function (c, b) {
                        var d = this;
                        void 0 === b && (b = !1);
                        var e = jc(c, function (h) {
                            var k = !W(h.partNum);
                            return {
                                data: k ? void 0 : d.Xb[h.type](h.data),
                                uh: k ? h.data : void 0,
                                page: h.partNum,
                                end: h.end,
                                L: h.L
                            }
                        }), f = this.isSync || b ? Infinity : 10;
                        e = kc(this.l, e, f);
                        var g = [e];
                        this.Qb.push(e);
                        return e(Xe(function (h) {
                            h = ke(d.l, di, {buffer: h});
                            h = kc(d.l, h, f, Le);
                            g.push(h);
                            d.Qb.push(h);
                            return h
                        }))(Xe(function (h) {
                            h = Lf(d.l, h.slice(-4));
                            h = kc(d.l, h,
                                f, Le);
                            g.push(h);
                            d.Qb.push(h);
                            return h
                        }))(Vg(function (h) {
                            h = h[h.length - 1];
                            z(function (k) {
                                k = ye(d.l)(k, d.Qb);
                                d.Qb.splice(k, 1)
                            }, g);
                            return h
                        }))
                    };
                    a.prototype.Ua = function (c) {
                        return ke(this.l, Of, this.Xb[c.type](c.data))(Me(B))
                    };
                    a.prototype.$a = function (c) {
                        return c[0]
                    };
                    a.prototype.kd = function (c, b) {
                        for (var d = Lf(this.l, c)(Me(B)), e = Math.ceil(d.length / b), f = [], g = 0; g < b; g += 1) f.push(d.slice(g * e, e * (g + 1)));
                        return f
                    };
                    a.prototype.isEnabled = function () {
                        return ci(this.l)
                    };
                    return a
                }(), bu = function () {
                    return function (a, c, b, d) {
                        var e =
                            this;
                        this.vd = this.ac = !1;
                        this.eb = [];
                        this.Uf = [];
                        this.qf = [];
                        this.send = function (f, g, h, k) {
                            f = e.sender(f, e.xc, g);
                            h && k && f.then(h, k);
                            return f
                        };
                        this.Pe = function (f, g, h, k) {
                            return new I(function (l, m) {
                                f.push([g, h, l, m, k])
                            })
                        };
                        this.xi = function () {
                            e.eb = Eg(function (h, k) {
                                return h[4].partNum - k[4].partNum
                            }, e.eb);
                            var f = M(function (h, k, l) {
                                k = k[4];
                                return h && l + 1 === k.partNum
                            }, !0, e.eb), g = !!e.eb[e.eb.length - 1][4].end;
                            return f && g
                        };
                        this.Jd = function (f) {
                            rh(e.l, f.slice(), function (g) {
                                e.send(g[0], g[1], g[2], g[3])
                            }, 20, "s.w2.sf.fes");
                            Ed(f)
                        };
                        this.Qh = function () {
                            e.vd || (e.vd = !0, e.Jd(e.Uf), e.Jd(e.qf))
                        };
                        this.sh = function (f) {
                            return M(function (g, h) {
                                var k = "page" === h.type && !h.M, l = "eof" === h.data.type, m = k && !!h.partNum;
                                return {Ad: g.Ad || m, zd: g.zd || k, yd: g.yd || l}
                            }, {zd: !1, yd: !1, Ad: !1}, f)
                        };
                        this.vi = function (f, g, h, k) {
                            k ? (f = e.Pe(e.eb, f, g, h[0]), e.xi() && (e.Jd(e.eb), e.ac = !0)) : (e.ac = !0, f = e.send(f, g));
                            return f
                        };
                        this.ri = function (f) {
                            var g;
                            return e.Ji ? (g = {}, g["wv-type"] = Ja(function (h) {
                                return "eof" === n(h, "data.type")
                            }, f) ? "2" : "8", g) : {}
                        };
                        this.Gf = function (f, g, h) {
                            g = {
                                G: e.ri(h),
                                H: Da(), Y: {fa: g}, Ta: {Zd: e.Gi}
                            };
                            f && g.H.C("bt", 1);
                            return g
                        };
                        this.Ih = function (f, g, h) {
                            f = e.Gf(!1, f, g);
                            return e.ac ? e.send(f, h) : e.Pe(e.qf, f, h, g)
                        };
                        this.Wi = function (f, g, h) {
                            f = e.Gf(!0, f, g);
                            if (e.ac) return e.send(f, h);
                            var k = e.sh(g), l = k.zd, m = k.yd;
                            k = k.Ad;
                            var p;
                            l && (p = e.vi(f, h, g, k));
                            e.vd ? l || (p = e.send(f, h)) : (l || (p = e.Pe(e.Uf, f, h, g)), (e.ac || m) && e.Qh());
                            return p
                        };
                        this.Ji = d;
                        this.l = a;
                        this.Gi = b;
                        this.sender = Ba(a, "W", c);
                        this.xc = c
                    }
                }(), rl = x(function (a) {
                    var c = H(a), b = c.o("isEU");
                    if (W(b)) {
                        var d = Ga(je(a, "is_gdpr") || "");
                        if (K(d, [0,
                            1])) c.C("isEU", d), b = !!d; else if (a = Ra(a).o("wasSynced"), a = n(a, "params.eu")) c.C("isEU", a), b = !!a
                    }
                    return b
                }, function (a) {
                    return H(a).o("isEU")
                }), xf = C("i.e", rl), cu = C("i.ep", function (a) {
                    rl(a)
                }), du = C("w2", function (a, c) {
                    function b() {
                        h = !0
                    }

                    var d = H(a), e = N(c);
                    if (!c.Kb || bd(a) || !a.MutationObserver || !Aa("Element", a.Element)) return B;
                    Aa("MutationObserver", a.MutationObserver) || Dd(a, e).warn("MutationObserver is overriden, webvisor is not guaranteed to work in this environment");
                    var f = za(function (k, l) {
                            ra(c, l)["catch"](k)
                        }),
                        g = cc(a)(Xe(E([a, c], Xm)))(Vg(function (k) {
                            return new Zt(a, k)
                        })), h = !1;
                    Eq([g, f])(Sa(D(a, "wv2.R.c"), function (k) {
                        var l = k[0];
                        k = k[1];
                        if (!h) {
                            b = function () {
                                h || (h = !0, l && l.stop())
                            };
                            var m = d.o("wv2Counter");
                            if (!Sh(a, k) || m) b(); else if (ia(a).D(a, ["beforeunload", "unload"], b), d.C("wv2Counter", e), d.C("stopRecorder", b), k = [new af(a)], k.unshift(new au(a)), k = bb(function (v) {
                                return v.isEnabled()
                            }, k)) {
                                m = new bu(a, c, !(k instanceof af), 0);
                                var p = ol.Bf(e, "m", F(m.Wi, m), k, a), q = ol.Bf(e, "e", F(m.Ih, m), k, a);
                                k = Ym();
                                m = k.bj;
                                q.D("ag", k.fh);
                                q.D("p", m);
                                p.D("see", function (v) {
                                    var w = !1;
                                    z(function (G) {
                                        "page" === G.type && (w = !0)
                                    }, v);
                                    w && (h || q.push({
                                        type: "event",
                                        data: {type: "fatalError", aa: {code: "invalid-snapshot", Jh: "p.s.f", stack: ""}}
                                    }), b())
                                });
                                var r = hb(function (v) {
                                    "eof" === n(v, "data.type") ? (q.push(v), p.push(v), q.flush(), p.flush()) : ("event" === v.type ? q : p).push(v)
                                });
                                V(a, b, 864E5);
                                Kb(a, function () {
                                    ib(a, {da: e, name: "webvisor", data: {version: 2}});
                                    l.start(r)
                                })
                            }
                        }
                    }));
                    return function () {
                        return b()
                    }
                }), eu = C("w2.cs", function (a, c) {
                    var b, d = N(c);
                    bg(a, d, (b = {}, b.webvisor =
                        !!c.Kb, b))
                }), sl = x(Ac, N), Oh = t(dd, oc), tl = B, fu = pb("isp.stat", function (a, c) {
                    return new I(function (b, d) {
                        if (rq(a, fl, "isp")) {
                            var e = function (f) {
                                ("1" === f ? b : d)();
                                tl();
                                f = zj(fl);
                                K("isp", f.zb) && (f.zb = Z(t(ka("isp"), Tb), f.zb), f.zb.length || (nc(f.qb), f.qb = null))
                            };
                            tl = ia(a).D(a, ["message"], E([c, e], D(a, "isp.stat.m", Wm)));
                            V(a, e, 1500)
                        } else d()
                    })
                }), gu = pb("isp", function (a, c) {
                    ra(c, function (b) {
                        var d = bb(function (g) {
                            return n(b, "settings." + g)
                        }, ["rt", "mf"]);
                        if (d && Qd(a)) {
                            var e = oi(b) && !be(a), f = sl(c);
                            y(f, {bd: d, status: e ? 3 : 4});
                            if (!e) return d =
                                Vm(a, c, d), e = function (g) {
                                f.status = g
                            }, fu(a, d).then(u(1, e))["catch"](u(2, e))
                        }
                    })["catch"](D(a, "l.isp"))
                }), ul = C("fbq.o", function (a, c, b) {
                    var d = n(a, "fbq");
                    if (d && d.callMethod) {
                        var e = function () {
                            var g = Ca(arguments), h = d.apply(void 0, g);
                            c(g);
                            return h
                        };
                        y(e, d);
                        b && z(c, b);
                        a.fbq = e
                    } else var f = V(a, E([a, c, xa(ya(d && d.queue))], ul), 1E3, "fbq.d");
                    return F(na, null, a, f)
                }), Yc, Ab, Zc,
                mh = (Yc = {}, Yc.add_to_wishlist = "add-to-wishlist", Yc.begin_checkout = "begin-checkout", Yc.generate_lead = "submit-lead", Yc.add_payment_info = "add-payment-info",
                    Yc),
                nh = (Ab = {}, Ab.AddToCart = "add-to-cart", Ab.Lead = "submit-lead", Ab.InitiateCheckout = "begin-checkout", Ab.Purchase = "purchase", Ab.CompleteRegistration = "register", Ab.Contact = "submit-contact", Ab.AddPaymentInfo = "add-payment-info", Ab.AddToWishlist = "add-to-wishlist", Ab.Subscribe = "subscribe", Ab),
                Tm = (Zc = {}, Zc["1"] = mh, Zc["2"] = mh, Zc["3"] = mh, Zc["0"] = nh, Zc),
                Um = [nh.AddToCart, nh.Purchase], hu = la(function (a, c) {
                    var b = n(c, "ecommerce"), d = n(c, "event") || "";
                    if (!(b = b && d && {version: "3", Dc: d})) a:{
                        if (ca(c) || Pa(c)) if (b = Ca(c), d =
                            b[1], "event" === b[0] && d) {
                            b = {version: "2", Dc: d};
                            break a
                        }
                        b = void 0
                    }
                    b || (b = (b = n(c, "ecommerce")) && {version: "1", Dc: J(",", da(b))});
                    b && a(b)
                }), iu = C("ag.e", function (a, c) {
                    var b = [], d = D(a, "ag.s", E([ha, b], z));
                    "0" === c.ba && ra(c, function (e) {
                        if (n(e, "settings.auto_goals") && Ha(a, c) && (e = te(a, c, "autogoal").reachGoal)) {
                            e = E([e, c.Hd], Sm);
                            var f = hu(e);
                            e = E([a, e], Rm);
                            b.push(ul(a, e));
                            b.push(Pi(a, "dataLayer", function (g) {
                                g.Aa.D(f)
                            }))
                        }
                    });
                    return d
                }), ju = /[^\d.,]/g, ku = /[.,]$/, Pm = C("ep.pp", function (a, c) {
                    if (!c) return 0;
                    a:{
                        var b = c.replace(ju,
                            "").replace(ku, "");
                        var d = "0" === b[b.length - 1];
                        for (var e = b.length; 0 < e && !(3 < b.length - e + 1 && d); --e) {
                            var f = b[e - 1];
                            if (K(f, [",", "."])) {
                                d = f;
                                break a
                            }
                        }
                        d = ""
                    }
                    b = d ? c.split(d) : [c];
                    d = d ? b[1] : "00";
                    b = parseFloat(Ob(b[0]) + "." + Ob(d));
                    d = Math.pow(10, 8) - .01;
                    a.isNaN(b) ? b = 0 : (b = a.Math.min(b, d), b = a.Math.max(b, 0));
                    return b
                }),
                lu = [[["EUR", "\u20ac"], "978"], [["USD", "\u0423\\.\u0415\\.", "\\$"], "840"], [["UAH", "\u0413\u0420\u041d", "\u20b4"], "980"], ["\u0422\u0413 KZT \u20b8 \u0422\u04a2\u0413 TENGE \u0422\u0415\u041d\u0413\u0415".split(" "),
                    "398"], [["GBP", "\u00a3", "UKL"], "826"], ["RUR RUB \u0420 \u0420\u0423\u0411 \u20bd P \u0420UB P\u0423\u0411 P\u0423B PY\u0411 \u0420Y\u0411 \u0420\u0423B P\u0423\u0411".split(" "), "643"]],
                mu = x(function (a) {
                    return new RegExp(a.join("|"), "i")
                }), nu = C("ep.cp", function (a) {
                    if (!a) return "643";
                    var c = Fi(a);
                    return (a = bb(function (b) {
                        return mu(b[0]).test(c)
                    }, lu)) ? a[1] : "643"
                }), ou = x(function () {
                    function a() {
                        var k = h + "0", l = h + "1";
                        f[k] ? f[l] ? (h = h.slice(0, -1), --g) : (e[l] = b(8), f[l] = 1) : (e[k] = b(8), f[k] = 1)
                    }

                    function c() {
                        var k = h +
                            "1";
                        f[h + "0"] ? f[k] ? (h = h.slice(0, -1), --g) : (h += "1", f[h] = 1) : (h += "0", f[h] = 1)
                    }

                    function b(k) {
                        void 0 === k && (k = 1);
                        var l = d.slice(g, g + k);
                        g += k;
                        return l
                    }

                    for (var d = Mh("Cy2FcreLJLpYXW3BXFJqldVsGMwDcBw2BGnHL5uj1TKstzse3piMo3Osz+EqDotgqs1TIoZvKtMKDaSRFztgUS8qkqZcaETgKWM54tCpTXjV5vW5OrjBpC0jF4mspUBQGd95fNSfv+vz+g+Hze33Hg8by+Yen1PP6zsdl7PQCwX9mf+f7FMb9x/Pw+v2Pp8Xy74eTwuBwTt913u4On1XW6hxOO5nIzFam00tC218S0kaeugpqST+XliLOlMoTpRQkuewUxoy4CT3efWtdFjSAAm+1BkjIhyeU4vGOf13a6U8wzNY4bGo6DIUemE7N3SBojDr7ezXahpWF022y8mma1NuTnZbq8XZZlPStejfG/CsbPhV6/bSnA==").join(""),
                             e = {}, f = {}, g = 1, h = ""; g < d.length - 1;) ("0" === b() ? c : a)();
                    return e
                }), Mm = C("ep.dec", function (a, c) {
                    if (!c || bd(a)) return [];
                    var b = Mh(c), d = b[1], e = b[2], f = b.slice(3);
                    if (2 !== Sg(b[0])) return [];
                    b = ou();
                    f = f.join("");
                    e = Sg(d + e);
                    var g = "";
                    d = "";
                    for (var h = 0; d.length < e && f[h];) g += f[h], b[g] && (d += String.fromCharCode(Sg(b[g])), g = ""), h += 1;
                    b = "";
                    for (f = 0; f < d.length;) e = d.charCodeAt(f), 128 > e ? (b += String.fromCharCode(e), f++) : 191 < e && 224 > e ? (g = d.charCodeAt(f + 1), b += String.fromCharCode((e & 31) << 6 | g & 63), f += 2) : (g = d.charCodeAt(f + 1), b += String.fromCharCode((e &
                        15) << 12 | (g & 63) << 6 | d.charCodeAt(f + 2) & 63), f += 3);
                    d = tb(a, b);
                    return ca(d) ? A(br, d) : []
                }), Om = C("ep.ent", function (a, c, b) {
                    a = "" + Va(a, 10, 99);
                    b = "" + 100 * c + b + a;
                    if (16 < Pa(b)) return "";
                    b = Nh("0", 16, b);
                    c = b.slice(0, 8);
                    b = b.slice(-8);
                    c = (+c ^ 92844).toString(35);
                    b = (+b ^ 92844).toString(35);
                    return "" + c + "z" + b
                }), vl = t(Lh, nu), wl = C("ep.ctp", function (a, c, b, d) {
                    var e = vl(a, b), f = Kh(a, d);
                    Jh(a, c, e, f);
                    Aa("MutationObserver", a.MutationObserver) && (new a.MutationObserver(function () {
                        var g = vl(a, b), h = Kh(a, d);
                        if (e !== g || f !== h) e = g, f = h, Jh(a, c, e, f)
                    })).observe(a.document.body,
                        {attributes: !0, childList: !0, subtree: !0, characterData: !0})
                }), pu = C("ep.chp", function (a, c, b, d, e) {
                    b && vf(a, c);
                    return d || e ? ia(a).D(a.document, ["click"], D(a, "ep.chp.cl", E([a, c, d, e], Nm))) : B
                }), tu = C("ep.i", function (a, c) {
                    return Id(a) ? Lm(a, c).then(function (b) {
                        var d = b.Eh, e = d[0], f = d[1], g = d[2], h = d[3], k = d[4], l = d[5], m = d[6], p = d[7],
                            q = d[8], r = d[9], v = d[10], w = d[11], G = d[12], Y = d[13], Q = d[14], oa = d[15];
                        if (!b.isEnabled) return I.resolve(B);
                        var ta = ee(a, e), vb = ee(a, h), ud = ee(a, m), se = ee(a, q),
                            qu = "" + e + f + g === "" + h + k + l;
                        return new I(function (ru,
                                               su) {
                            cc(a)(Sa(su, function () {
                                ta && wl(a, c, f, g, v, w, G);
                                vb && !qu && wl(a, c, k, l, Y, Q, oa);
                                ru(pu(a, c, ud || se, p, r))
                            }))
                        })
                    }) : I.resolve(B)
                }), tm = ["RTCPeerConnection", "mozRTCPeerConnection", "webkitRTCPeerConnection"],
                uu = C("cc.i", function (a, c) {
                    var b = E([a, c], Km);
                    b = E([a, b, 300, void 0], V);
                    ra(c, b)
                }), vu = u("9-d5ve+.r%7", O), wu = C("ad", function (a, c) {
                    if (!c.xb) {
                        var b = H(a);
                        if (!b.o("adBlockEnabled")) {
                            var d = function (m) {
                                K(m, ["2", "1"]) && b.C("adBlockEnabled", m)
                            }, e = ac(a), f = e.o("isad");
                            if (f) d(f); else {
                                var g = u("adStatus", b.C), h = function (m) {
                                    m =
                                        m ? "1" : "2";
                                    d(m);
                                    g("complete");
                                    e.C("isad", m, 1200);
                                    return m
                                }, k = Ba(a, "adb", c);
                                if (!b.o("adStatus")) {
                                    g("process");
                                    var l = "metrika/a" + vu().replace(/[^a-v]+/g, "") + "t.gif";
                                    Hm(a, function () {
                                        return k({ja: {ta: l}}).then(u(!1, h))["catch"](u(!0, h))
                                    })
                                }
                            }
                        }
                    }
                }), xu = C("pr.p", function (a, c) {
                    var b, d;
                    if (vg(a)) {
                        var e = Ba(a, "5", c), f = Da((b = {}, b.pq = 1, b.ar = 1, b));
                        e({
                            H: f,
                            G: (d = {}, d["page-url"] = S(a).href, d["page-ref"] = n(a, "document.referrer") || "", d)
                        }, c)["catch"](D(a, "pr.p.s"))
                    }
                }), Ih = !1, yu = C("fid", function (a) {
                    var c, b = B;
                    if (!T(a.PerformanceObserver)) return b;
                    var d = H(a);
                    if (d.o("fido")) return b;
                    d.C("fido", !0);
                    var e = new a.PerformanceObserver(D(a, "fid", function (f) {
                        f = f.getEntries()[0];
                        d.C("fid", a.Math.round(100 * (f.processingStart - f.startTime)));
                        b()
                    }));
                    b = function () {
                        return e.disconnect()
                    };
                    try {
                        e.observe((c = {}, c.type = "first-input", c.buffered = !0, c))
                    } catch (f) {
                    }
                    return b
                }), Gh = {
                    1882689622: 1,
                    2318205080: 1,
                    3115871109: 1,
                    3604875100: 1,
                    339366994: 1,
                    2890452365: 1,
                    849340123: 1,
                    173872646: 1,
                    2343947156: 1,
                    655012937: 1,
                    3724710748: 1,
                    3364370932: 1,
                    1996539654: 1,
                    2065498185: 1,
                    823651274: 1,
                    12282461: 1,
                    1555719328: 1,
                    1417229093: 1,
                    138396985: 1,
                    3015043526: 1
                }, zu = C("p.sci", function (a, c) {
                    var b = H(a);
                    return b.o("scip") ? I.resolve() : ra(c, O).then(function (d) {
                        d = n(d, "settings.ins");
                        return !b.o("scip") && d ? Fm(a, c, b) : null
                    }, D(a, "ins.cs"))
                }), Au = C("lt.p", pb("lt.p", function (a) {
                    var c;
                    if (Aa("PerformanceObserver", a.PerformanceObserver)) {
                        var b = 0, d = new a.PerformanceObserver(D(a, "lt.o", function (e) {
                            e && e.getEntries && (e = e.getEntries(), b = M(function (f, g) {
                                return f + g.duration
                            }, b, e), Nd(a).C("lt", b))
                        }));
                        try {
                            d.observe((c =
                                {}, c.type = "longtask", c.buffered = !0, c))
                        } catch (e) {
                            return B
                        }
                        return function () {
                            return d.disconnect()
                        }
                    }
                    return B
                })), Bu = x(t(X("performance.memory.jsHeapSizeLimit"), qa("concat", ""))),
                Cu = ["availWidth", "availHeight", "availTop"],
                Du = "appName vendor deviceMemory hardwareConcurrency maxTouchPoints appVersion productSub appCodeName vendorSub".split(" "),
                Eu = ["webgl", "experimental-webgl"], Em = [-.2, -.9, 0, .4, -.26, 0, 0, .732134444, 0],
                rf = u(Ta("ccf"), Xa),
                Dm = "prefers-reduced-motion;prefers-reduced-transparency;prefers-color-scheme: dark;prefers-color-scheme: light;pointer: none;pointer: coarse;pointer: fine;any-pointer: none;any-pointer: coarse;any-pointer: fine;scan: interlace;scan: progressive;color-gamut: srgb;color-gamut: p3;color-gamut: rec2020;update: fast;update: slow;update: none;grid: 0;grid: 2;hover: hover;inverted-colors: inverted;inverted-colors: none".split(";"),
                Eh = "video/ogg video/mp4 video/webm audio/x-aiff audio/x-m4a audio/mpeg audio/aac audio/wav audio/ogg audio/mp4".split(" "),
                Bm = "theora vorbis 1 avc1.4D401E mp4a.40.2 vp8.0 mp4a.40.5".split(" "), vm = x(Ai), Dh = x(tb, yb),
                Fu = C("phc.p", function (a, c) {
                    return Yk(a) ? B : ra(c, function (b) {
                        var d = c.id, e = Dc(a, void 0, d), f = e.o("phc_settings") || "";
                        if (b = n(b, "settings.phchange")) {
                            var g = mb(a, b) || "";
                            g !== f && e.C("phc_settings", g)
                        } else f && (b = Dh(a, f));
                        e = n(b, "clientId");
                        f = n(b, "orderId");
                        b = n(b, "phones") || [];
                        e && f && b.length && (f = {
                            Pb: "",
                            $b: "", wg: 0, qa: {}, Ba: [], Pf: !1, ob: !0, l: a, xc: c
                        }, y(f, {Pf: !0}), Ch(a, d, f), b = wd(a), e = Ci(a, b, 1E3), d = F(Ch, null, a, d, f), e.D(d), Di(a, b))
                    })
                }), oh = x(function (a, c) {
                    var b = H(a), d = Ra(a), e = [], f = E([a, c, b, d], cp);
                    pd(a) || dp(a, "14.1") || e.push(E([sm, "pp", ""], f));
                    var g = $k(a) && !pf(a, 68);
                    g || e.push(E([um, "pu", ""], f));
                    g || d.$d || Qd(a) || (e.push(E([rm, "zzlc", "na"], f)), e.push(E([qm, "cc", ""], f)));
                    return e.length ? {
                        Ea: function (h, k) {
                            if (0 === b.o("isEU")) try {
                                z(Qi, e)
                            } catch (l) {
                            }
                            k()
                        }, Z: function (h, k) {
                            var l = h.H;
                            if (l && 0 === b.o("isEU")) try {
                                z(za(l),
                                    e)
                            } catch (m) {
                            }
                            k()
                        }
                    } : {}
                }, function (a, c) {
                    return N(c)
                }), Gu = t(function (a) {
                    a = n(a, "navigator.plugins") || [];
                    return Pa(a) ? t(ya, Na, Ar(function (c, b) {
                        return c.name > b.name ? 1 : 2
                    }), hb($o))(a) : ""
                }, qd(",")), Hu = function (a) {
                    return function (c) {
                        var b = ab(c);
                        if (!b) return "";
                        b = b("canvas");
                        var d = [], e = a(), f = e.Lh;
                        e = e.Bh;
                        try {
                            var g = qa("getContext", b);
                            d = A(t(O, g), e)
                        } catch (h) {
                            return ""
                        }
                        return (g = bb(O, d)) ? f(c, {canvas: b, oh: g}) : ""
                    }
                }(function () {
                    return {Bh: Eu, Lh: mm}
                }), km = ["name", "lang", "localService", "voiceURI", "default"], Iu = C("p.tfs", function (a) {
                    var c =
                        H(a);
                    if (!c.o("tfs")) {
                        c.C("tfs", !0);
                        c = ia(a);
                        var b = B;
                        b = c.D(a, ["message"], function (d) {
                            try {
                                var e = d.origin
                            } catch (h) {
                            }
                            if (e && "https://iframe-toloka.com" === e && (d = tb(a, d.data), La(d) && "appendremote" === d.action)) if (d = Ra(a), d.o("tfsc") || a.confirm("\u0414\u043e\u043f\u043e\u043b\u043d\u0435\u043d\u0438\u0435 \u201c\u0420\u0430\u0437\u043c\u0435\u0442\u043a\u0430 \u0441\u0430\u0439\u0442\u043e\u0432\u201c \u043e\u0442 toloka.ai \u0437\u0430\u043f\u0440\u0430\u0448\u0438\u0432\u0430\u0435\u0442 \u0434\u043e\u0441\u0442\u0443\u043f \u043a \u0441\u043e\u0434\u0435\u0440\u0436\u0438\u043c\u043e\u043c\u0443 \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u044b. \u0420\u0430\u0437\u0440\u0435\u0448\u0438\u0442\u044c?")) {
                                d.C("tfsc",
                                    1);
                                var f, g;
                                H(a).C("_u", (f = {}, f.getCachedTags = Uf, f.button = (g = {}, g.closest = u(a, Vf), g.select = Wf, g.getData = u(a, Xf), g), f));
                                lc(a, {src: "https://yastatic.net/s3/metrika/2.1540128042.1/form-selector/button_ru.js"});
                                b()
                            } else a.close()
                        })
                    }
                }), Ju = Ya(/[a-z\u0430-\u044f,.]/gi), Ku = C("ice", function (a, c, b) {
                    var d = Ha(a, c);
                    if (d) {
                        var e = n(b, "target");
                        if (e && (c = n(e, "value"), (c = ob(c)) && !(100 <= Pa(c)))) {
                            b = Ob(c);
                            var f = 0 < c.indexOf("@"), g = "tel" === n(e, "type") || !f && Pa(b);
                            if (f || g) {
                                if (g) {
                                    if (Ju(c)) return;
                                    g = c[0];
                                    var h = b[0];
                                    if (g !== h &&
                                        "+" !== g) return;
                                    var k = c[1];
                                    if ("+" === g && k !== h) return;
                                    c = c[Pa(c) - 1];
                                    g = b[Pa(b) - 1];
                                    if (c !== g) return;
                                    c = b
                                }
                                b = f ? 5 : 11;
                                g = f ? 100 : 16;
                                Pa(c) < b || Pa(c) > g || fj(a, c).then(function (l) {
                                    var m, p, q, r = qg(a, e);
                                    d.params((m = {}, m.__ym = (p = {}, p.fi = ug((q = {}, q.a = f ? 1 : 0, q.b = r, q.c = l, q)).Ca(), p), m))
                                }, D(a, "ice.s"))
                            }
                        }
                    }
                }), Lu = ["text", "email", "tel"], Mu = ["cc-", "name", "shipping"], Nu = C("icei", function (a, c) {
                    if (Id(a) && Vk(a)) {
                        var b = !1, d = [];
                        cc(a)(Sa(B, E([c, function (e) {
                            var f = n(e, "settings.cf");
                            e = xf(a) || n(e, "settings.eu");
                            if (f && !e && !b) {
                                var g = ia(a);
                                f =
                                    xb("input", a.document.body);
                                z(function (h) {
                                    Gf(a, h) || !K(h.type, Lu) || Ja(O, A(u(h.autocomplete, pc), Mu)) || d.push(g.D(h, ["blur"], E([a, c], Ku)))
                                }, f)
                            }
                        }], ra)));
                        return function () {
                            z(ha, d);
                            b = !0
                        }
                    }
                }), zh, Ou = C("p.ai", function (a, c) {
                    return new I(function (b) {
                        (pd(a) || jf(a)) && b(ra(c, function (d) {
                            var e;
                            return (d = n(d, "settings.sbp")) ? yh(a, y({}, d, (e = {}, e.c = c.id, e)), 10) : B
                        }));
                        b(B)
                    })
                }), Pu = "architecture bitness model platformVersion uaFullVersion fullVersionList".split(" "),
                xl = pb("uah", function (a) {
                    if (!Aa("getHighEntropyValues",
                        n(a, "navigator.userAgentData.getHighEntropyValues"))) return I.reject("0");
                    try {
                        return a.navigator.userAgentData.getHighEntropyValues(Pu).then(function (c) {
                            if (!La(c)) throw "2";
                            return c
                        }, function () {
                            throw "1";
                        })
                    } catch (c) {
                        return I.reject("3")
                    }
                }),
                yl = new RegExp(J("|", "yandex.com/bots;Googlebot;APIs-Google;Mediapartners-Google;AdsBot-Google;FeedFetcher-Google;Google-Read-Aloud;DuplexWeb-Google;Google Favicon;googleweblight;Lighthouse;Mail.RU_Bot;StackRambler;Slurp;msnbot;bingbot;www.baidu.com/search/spi_?der.htm".split(";")).replace(/[./]/g,
                    "\\$&")), am = x(function (a) {
                    var c = gb(a);
                    return (c = yl.test(c)) ? I.resolve(c) : xl(a).then(function (b) {
                        try {
                            return M(function (d, e) {
                                return d || yl.test(e.brand)
                            }, !1, b.brands)
                        } catch (d) {
                            return !1
                        }
                    }, u(!1, O))
                }), Xb = ["0", "1", "2", "3"], Ec = Xb[0], hf = Xb[1], Qu = Xb[2],
                mf = A(t(O, qa("concat", "GDPR-ok-view-detailed-")), Xb),
                ae = xa("GDPR-ok GDPR-cross GDPR-cancel 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 GDPR-settings GDPR-ok-view-default GDPR-ok-view-detailed 21 22 23".split(" "), mf, ["28", "29", "30"]),
                Ru = "3 13 14 15 16 17 28".split(" "),
                lf = t(hb(X("ymetrikaEvent.type")), Rg(uc(ae))), hm = {
                    url: "https://yastatic.net/s3/gdpr/popup/v2/",
                    Sf: "ru en et fi lt lv pl fr no sr".split(" "),
                    ag: "_two_main_buttons"
                }, Wb,
                gm = (Wb = {}, Wb["GDPR-ok"] = "ok", Wb["GDPR-ok-view-default"] = "ok-default", Wb["GDPR-ok-view-detailed"] = "ok-detailed", Wb["GDPR-ok-view-detailed-0"] = "ok-detailed-all", Wb["GDPR-ok-view-detailed-1"] = "ok-detailed-tech", Wb["GDPR-ok-view-detailed-2"] = "ok-detailed-tech-analytics", Wb["GDPR-ok-view-detailed-3"] = "ok-detailed-tech-other", Wb),
                ce = [], uh =
                    t(lf, hb(zr(ae)), qd(",")), vh = t(Rg(uc(ae)), Pa, Boolean), Zl = x(function (a, c) {
                    var b = c.o("gdpr");
                    return K(b, Xb) ? "-" + b : ""
                }), ad = {}, zl = x(Ac), Wl = t(qa("exec", /counterID=(\d+)/), X("1")), Al = la(function (a, c) {
                    var b = zl(a), d = ya(c), e = d[0], f = d[1], g = d.slice(2);
                    if (f) {
                        d = Vl(a, e);
                        var h = d[0], k = d[1];
                        d = N(k);
                        b[d] || (b[d] = {});
                        b = b[d];
                        c.tf || ad[f] && M(function (l, m) {
                            return l || !!m(a, k, g, h)
                        }, !1, ad[f]) || ("init" === f ? (c.tf = !0, h ? Db(a, "" + e, "Duplicate counter " + e + " initialization") : a["yaCounter" + k.id] = new a.Ya[sa.uc](y({}, g[0], k))) : h && h[f] &&
                        b.Ei ? (h[f].apply(h, g), c.tf = !0) : (d = b.xg, d || (d = [], b.xg = d), d.push(xa([e, f], g))))
                    }
                }), Su = pb("is", function (a) {
                    var c = ac(a);
                    if (Ae(a, "0")) c.Rb("debug_build"); else {
                        var b = Ae(a, "2"), d = c.o("debug_build") === sa.jb;
                        if (b || d) return c.C("debug_build", sa.jb), lc(a, {src: el + "/tag_debug.js"})
                    }
                });
            "function" == typeof Promise && Promise.resolve();
            Na([void 0, void 0, void 0, void 0, void 0, void 0, void 0, void 0, void 0]);
            var Bl = x(vd), Tu = x(function () {
                    var a = M(function (c, b) {
                        "ru" !== b && (c[b] = dl + "." + b);
                        return c
                    }, {}, ng);
                    z(function (c) {
                        a[c] =
                            c
                    }, da(gl));
                    return a
                }), Ql = x(function (a) {
                    a = S(a).hostname;
                    return (a = bb(t(X("1"), Bi("test"), Ib(ha)(a)), pa(gl))) && a[0]
                }), Cl = function (a, c) {
                    return function (b, d) {
                        var e = N(d);
                        e = Tu(e);
                        var f = Ol(b, e), g = H(b), h = eb(b);
                        return Qd(b) || Kd(b) ? {} : {
                            Z: function (k, l) {
                                var m = k.H, p = sh(b);
                                m = !(m && m.o("pv"));
                                if (!p || h || m || !f.length) return l();
                                if (g.o("startSync")) Bl(b).push(l); else {
                                    g.C("startSync", !0);
                                    p = E([b, f, B, !1], a);
                                    m = kf[0];
                                    if (!m) return l();
                                    m(b).then(p).then(l, t(gd(l), D(b, c)))["catch"](B)
                                }
                            }
                        }
                    }
                }(function (a, c, b, d) {
                    var e = fa(a), f = H(a),
                        g = Ra(a);
                    b = ed(a, "c");
                    var h = Eb(a, b);
                    return M(function (k, l) {
                        function m() {
                            var r = g.o("synced");
                            f.C("startSync", !1);
                            r && (r[l.Ri] = p, g.C("synced", r));
                            r = Bl(a);
                            z(ha, r);
                            Ed(r)
                        }

                        var p, q = h({
                            Y: {
                                Fa: ["sync.cook"],
                                Ib: 1500
                            }
                        }, [sa.Za + "//" + l.Fj + "/sync_cookie_image_check" + (d ? "_secondary" : "")]).then(function () {
                            p = e(lb);
                            m()
                        })["catch"](function () {
                            p = e(lb) - 1435;
                            m()
                        });
                        q = u(q, O);
                        return k.then(q)
                    }, I.resolve(), c)["catch"](D(a, "ctl"))
                }, "sy.c"), Bb,
                Ml = (Bb = {}, Bb.brands = "chu", Bb.architecture = "cha", Bb.bitness = "chb", Bb.uaFullVersion = "chf",
                    Bb.fullVersionList = "chl", Bb.mobile = "chm", Bb.model = "cho", Bb.platform = "chp", Bb.platformVersion = "chv", Bb),
                Uu = pb("ot", function (a, c) {
                    return ia(a).D(a, ["message"], C("ot", u(E([a, c], dd), Jl)))
                }), Il = C("destruct.e", function (a, c, b) {
                    return function () {
                        var d = H(a), e = c.id;
                        z(function (f, g) {
                            return T(f) && D(a, "dest.fr." + g, f)()
                        }, b);
                        delete d.o("counters")[N(c)];
                        delete a["yaCounter" + e]
                    }
                }), $c = H(window);
            $c.Va("hitParam", {});
            $c.Va("lastReferrer", window.location.href);
            (function () {
                U.push(function (a, c) {
                    var b;
                    return b = {}, b.firstPartyParams =
                        ms(a, c), b.firstPartyParamsHashed = Qp(a, c), b
                });
                Ue.push("fpp", "fpmh")
            })();
            (function () {
                var a = H(window);
                a.Va("getCounters", ns(window));
                Cc.push(os);
                Ug.push(function (c, b) {
                    b.counters = a.o("getCounters")
                })
            })();
            (function () {
                U.push(function (a, c) {
                    ib(a, {da: N(c), name: "counter", data: c})
                })
            })();
            Ea["1"] = kb;
            U.push(ps);
            va["1"] = Ye;
            zb(aj, -1);
            Qb["1"] = [[aj, -1], [Ke, 1], [De, 2], [Hb(), 3]];
            U.push(qs);
            U.push(C("p.ar", function (a, c) {
                var b, d = Ba(a, "a", c);
                return b = {}, b.hit = function (e, f, g, h, k, l) {
                    var m, p, q = {
                        G: {}, H: Da((m = {}, m.pv = 1, m.ar =
                            1, m))
                    };
                    if (e) return f = La(f) ? {
                        title: f.title,
                        eg: f.referer,
                        ea: f.params,
                        pc: f.callback,
                        l: f.ctx
                    } : {
                        title: f,
                        eg: g,
                        ea: h,
                        pc: k,
                        l: l
                    }, g = Cd(c), g.url !== e && (g.ref = g.url, g.url = e), e = e || S(a).href, g = f.eg || g.ref || a.document.referrer, h = Gb(a, c, "PageView. Counter " + c.id + ". URL: " + e + ". Referrer: " + g, f.ea), k = y(q.V || {}, {
                        ea: f.ea,
                        title: f.title
                    }), q = d(y(q, {
                        V: k,
                        G: y(q.G || {}, (p = {}, p["page-url"] = e, p["page-ref"] = g, p))
                    }), c).then(h), Lc(a, "p.ar.s", q, f.pc || B, f.l)
                }, b
            }));
            Ea.a = kb;
            Qb.a = Rb;
            va.a = Ye;
            U.push(te);
            Ea.g = kb;
            va.g = Ye;
            Qb.g = Rb;
            U.push(rs);
            U.push(ss);
            Qb.t = Rb;
            Ea.t = kb;
            va.t = Wc;
            U.push(us);
            Qb["2"] = Rb;
            Ea["2"] = kb;
            va["2"] = Wc;
            Ea.r = wg("r");
            va.r = Ye;
            Cb.push(C("p.r", function (a, c) {
                var b = ws(a), d = Ba(a, "r", c), e = D(a, "rts.p");
                return ra(c, E([function (f, g) {
                    var h = {id: g.yh, ba: g.ba},
                        k = {Y: {fa: g.nj}, H: Da(g.lh), G: g.ea, V: {cc: g.cc}, ja: {ta: g.ta}};
                    g.La && (k.La = ug(g.La));
                    h = d(k, h)["catch"](e);
                    return f.then(u(h, O))
                }, I.resolve(), b], M))["catch"](e)
            }));
            ba("r", function (a) {
                return {
                    Z: function (c, b) {
                        var d = c.H, e = void 0 === d ? Da() : d, f = c.V.cc, g = Bd(a);
                        d = e.o("rqnl", 0) + 1;
                        e.C("rqnl",
                            d);
                        if (e = n(g, J(".", [f, "browserInfo"]))) e.rqnl = d, $f(a);
                        b()
                    }, Ea: function (c, b) {
                        Wi(a, c);
                        b()
                    }
                }
            }, 1);
            zb(ue, 100);
            ba("1", ue, 100);
            U.push(xs);
            ba("n", Ke, 1);
            ba("n", De, 2);
            ba("n", Hb(), 3);
            ba("n", ue, 100);
            Ea.n = kb;
            va.n = Wc;
            fc({df: {ia: "accurateTrackBounce"}});
            U.push(ys);
            Ea.m = wg("cm");
            va.m = hs;
            ba("m", Hb(["u", "v", "vf"]), 1);
            ba("m", ue, 2);
            fc({vh: {ia: "clickmap"}});
            U.push(zs);
            U.push(As);
            U.push(Cs);
            U.push(Ds);
            (function () {
                U.push(Es);
                Ue.push("ecommerce");
                fc({
                    Hd: {
                        ia: "ecommerce", bb: function (a) {
                            if (a) return !0 === a ? "dataLayer" : "" + a
                        }
                    }
                })
            })();
            U.push(Fs);
            Cb.push(Hs);
            U.push(Is);
            Ue.push("user_id");
            Cc.push(C("p.st", Js));
            U.push(Ks);
            zb(function (a, c) {
                return {
                    Ea: function (b, d) {
                        var e = Ha(a, c);
                        e = e && e.userParams;
                        var f = (b.V || {}).Xe;
                        e && f && e(f);
                        d()
                    }
                }
            }, 0);
            Xd.push(ks);
            U.push(Ns);
            U.push(Os);
            We.push(function (a) {
                var c = H(a);
                c.o("i") || (c.C("i", !0), ia(a).D(a, ["message"], u(a, hp)))
            });
            (function () {
                var a, c = (a = {}, a.tp = t(yb, Xj, Fb), a.tpid = t(yb, Qq), a);
                y(Ee, c)
            })();
            zb(sd, 20);
            ba("n", sd, 20);
            ba("1", sd, 20);
            (function () {
                var a;
                Ri.push("impressions", "click", "promoView", "promoClick");
                var c = (a = {}, a.promotion_name = "name", a.promotion_id = "id", a.item_id = "product_id", a.item_name = "product_name", a);
                yd.view_item_list = {event: "impressions", za: vc};
                yd.select_item = {event: "click", Ma: "products", za: c};
                yd.view_promotion = {event: "promoView", Ma: "promotions", za: c};
                yd.select_promotion = {event: "promoClick", Ma: "promotions", za: c}
            })();
            U.push(function (a, c) {
                var b;
                return b = {}, b.ecommerceAdd = C("ecm.a", Ps(a, c)), b.ecommerceRemove = C("ecm.r", Qs(a, c)), b.ecommerceDetail = C("ecm.d", Rs(a, c)), b.ecommercePurchase = C("ecm.p",
                    Ss(a, c)), b
            });
            (function () {
                var a, c = {};
                c.bu = $s;
                c.pri = Qo;
                c.wv = u(2, O);
                c.ds = To;
                c.co = function (b) {
                    return hd(H(b).o("jn"))
                };
                c.td = bt;
                y(c, (a = {}, a.iss = t(Lr, Fb), a.hdl = t(Mr, Fb), a.iia = t(Nr, Fb), a.cpf = t(Zs, Fb), a.ntf = x(function (b) {
                    a:switch (n(b, "Notification.permission")) {
                        case "denied":
                            b = !1;
                            break a;
                        case "granted":
                            b = !0;
                            break a;
                        default:
                            b = null
                    }
                    return Ua(b) ? null : b ? 2 : 1
                }), a.eu = Jb("isEU"), a.ns = Rk, a.np = function (b) {
                    if (Va(b, 0, 100)) b = null; else {
                        b = ob(Qf(b), 100);
                        for (var d = [], e = 0; e < b.length; e++) {
                            var f = b.charCodeAt(e);
                            128 > f ? d.push(f) :
                                (127 < f && 2048 > f ? d.push(f >> 6 | 192) : (d.push(f >> 12 | 224), d.push(f >> 6 & 63 | 128)), d.push(f & 63 | 128))
                        }
                        b = Rh(d)
                    }
                    return b
                }, a));
                y(Ee, c)
            })();
            (function () {
                var a = {};
                a.hc = Jb("hc");
                a.oo = Jb("oo");
                a.pmc = Jb("cmc");
                a.lt = function (c) {
                    var b = Nd(c).o("lt", null);
                    return b ? c.Math.round(100 * b) : b
                };
                a.re = t(mq, Fb);
                a.aw = function (c) {
                    c = bb(t(ma, Tb), [c.document.hidden, c.document.msHidden, c.document.webkitHidden]);
                    return ma(c) ? null : hd(!c)
                };
                a.yu = function (c) {
                    var b = S(c).hostname;
                    return K(b, ["dzen.ru", "ya.ru"]) ? (Dc(c, "").o("yandexuid") || "").substring(0,
                        25) : null
                };
                a.ifc = Jb("ifc");
                a.ifb = Jb("ifb");
                a.ecs = Jb("ecs");
                a.csi = Jb("scip");
                y(Gd, a)
            })();
            va.er = Xc;
            (function (a) {
                try {
                    var c = ed(a, "er"), b = Mo(a, c);
                    Vj.push(function (d, e, f, g) {
                        var h, k, l, m, p;
                        .01 >= a.Math.random() || b((h = {}, h[d] = (k = {}, k[sa.jb] = (l = {}, l[e] = (m = {}, m[f] = g ? (p = {}, p[a.location.href] = g, p) : a.location.href, m), l), k), h))
                    })
                } catch (d) {
                }
            })(window);
            (function () {
                Xd.push(Po);
                Ce.unshift(Lo);
                dh.push(function (a) {
                    var c = void 0;
                    void 0 === c && (c = !0);
                    H(a).C("oo", c)
                })
            })();
            zb(function (a, c) {
                return {
                    Z: function (b, d) {
                        var e = b.G, f = b.H;
                        !hl[c.id] && f.o("pv") && c.exp && !e.nohit && (e.exp = c.exp, hl[c.id] = !0);
                        d()
                    }
                }
            }, -99);
            U.push(gt);
            Qb.e = Rb;
            Ea.e = kb;
            va.e = Wc;
            fc({exp: {ia: "experiments"}});
            kk.experiments = "ex";
            (function () {
                var a;
                kf.push(ht);
                Ea.f = kb;
                y(va, (a = {}, a.f = Uk, a));
                ba("f", Hb(), 1);
                ba("f", oj, 2);
                ba("f", sd, 20)
            })();
            Xd.push(function (a, c) {
                var b = {da: N(c), Cd: Ha(a, c), Eg: fa(a), ke: Ra(a)}, d = b.Eg(lb);
                if (!b.ke.$d) {
                    var e = b.ke.o("ymoo" + b.da);
                    if (e && 30 > d - e) b = b.da, delete H(a).o("counters", {})[b], Xa(Ta("uws")); else ra(c, it(b))["catch"](D(a, "d.f"))
                }
            });
            (function () {
                var a,
                    c, b = [wb];
                y(va, (a = {}, a.s = b, a.S = b, a.u = Xc, a));
                y(Ea, (c = {}, c.s = Eb, c.S = kb, c.u = Eb, c));
                ba("s");
                ba("u");
                ba("S", Hb(["v", "hid", "u", "vf", "rn"]), 1);
                U.push(C("s", yo))
            })();
            Ea["8"] = Eb;
            va["8"] = [hg];
            Sk.push([hg, 0]);
            U.push(C("p.us", function (a, c) {
                return ra(c, function (b) {
                    var d, e = n(b, "settings.sbp");
                    return e && mi(a, b, {kb: c, bd: "8", data: y({}, e, (d = {}, d.c = c.id, d)), ie: "cs"})
                })
            }));
            ba("p", Hb(eh), 1);
            Ea.p = function (a, c, b) {
                return function (d, e) {
                    var f, g = y({H: Da()}, d);
                    g.G || (g.G = {});
                    var h = g.G, k = g.Ta;
                    k = void 0 === k ? {} : k;
                    h["wv-hit"] = h["wv-hit"] ||
                        "" + Lb(a);
                    h["page-url"] = h["page-url"] || a.location.href;
                    h.wmode = "0";
                    h["wv-part"] = "0";
                    h["page-url"] = h["page-url"] || a.location.href;
                    h["wv-type"] || (h["wv-type"] = k.Zd ? "5" : "4");
                    h = {
                        ja: {ta: "webvisor/" + e.id},
                        Y: y(g.Y, {Cb: (f = {}, f["Content-Type"] = "text/plain", f), Ze: "POST"}),
                        G: h
                    };
                    f = Oa(Pf(a, "pub", e), b);
                    return ne(a, c, f)(y(g, h))
                }
            };
            va.p = Na([0, wb]);
            Cb.push(nt);
            fc({Kb: {ia: "webvisor", bb: Boolean}, Dh: {ia: "disableFormAnalytics", bb: Boolean}});
            ba("4", Hb(eh), 1);
            Ea["4"] = li;
            va["4"] = Na([0, wb, Nc]);
            Cb.push(tt);
            (function () {
                ba("W",
                    Hb(eh), 1);
                va.W = Na([0, wb]);
                Ea.W = li;
                Cb.push(du);
                U.push(eu);
                fc({Kb: {ia: "webvisor"}});
                fc({Qj: {ia: "trustedDomains"}, sc: {ia: "childIframe", bb: Boolean}});
                dh.push(function (a) {
                    H(a).o("stopRecorder", B)()
                });
                Mk("wv")
            })();
            U.push(gu);
            ba("pi");
            Ea.pi = Eb;
            va.pi = Xc;
            Mk("w", function (a, c) {
                return {
                    Z: function (b, d) {
                        if (b.H) {
                            var e = sl(c), f = e.status;
                            "rt" === e.bd && f && (b.H.C("rt", f), b.ja || (b.ja = {}), b.ja.Bi = 1 === f ? Oh(a, c) + "." : "")
                        }
                        d()
                    }
                }
            }, 2);
            U.push(iu);
            U.push(tu);
            va["6"] = Na([0, wb]);
            Ea["6"] = Eb;
            U.push(uu);
            U.push(ct);
            (function () {
                Ug.push(function (a,
                                  c) {
                    c.informer = Im(a)
                })
            })();
            zb(tf, 6);
            ba("1", tf, 6);
            ba("adb");
            ba("n", tf, 4);
            va.adb = Xc;
            Ea.adb = ne;
            Cc.push(wu);
            va["5"] = Wc;
            Ea["5"] = kb;
            Qb["5"] = Z(t(xc, uc([Ke, De]), Tb), Rb);
            U.push(xu);
            ba("5", sd, 20);
            zb(Hh, 7);
            ba("n", Hh, 6);
            Cb.push(yu);
            Ea.d = kb;
            ba("d", Hb(["hid", "u", "v", "vf"]), 1);
            va.d = Xc;
            ba("n", function (a, c) {
                return {
                    Ea: function (b, d) {
                        if (!b.V || !b.V.force) {
                            var e = .002, f = c.id === sa.Zg ? 1 : .002, g, h, k, l, m;
                            void 0 === e && (e = 1);
                            void 0 === f && (f = 1);
                            var p = Rf(a);
                            if (p && T(p.getEntriesByType) && (e = Math.random() > e, f = Math.random() > f, !e || !f)) {
                                p =
                                    p.getEntriesByType("resource");
                                for (var q = {}, r = {}, v = {}, w = Gm(), G = S(a).href, Y = 0; Y < p.length; Y += 1) {
                                    var Q = p[Y], oa = Q.name.replace(/^https?:\/\//, "").split("?")[0], ta = oc(oa),
                                        vb = (g = {}, g.dns = Math.round(Q.domainLookupEnd - Q.domainLookupStart), g.tcp = Math.round(Q.connectEnd - Q.connectStart), g.duration = Math.round(Q.duration), g.response = Math.round(Q.responseEnd - Q.requestStart), g);
                                    "script" !== Q.initiatorType || e || (r[oa] = y(vb, (h = {}, h.name = Q.name, h.decodedBodySize = Q.decodedBodySize, h)));
                                    !Gh[ta] && !w[ta] || q[oa] || f || (q[oa] =
                                        y(vb, (k = {}, k.pages = G, k)))
                                }
                                da(q).length && (v.timings8 = q);
                                da(r).length && (v.scripts = r);
                                if (da(v).length) Ba(a, "d", c)({
                                    H: Da((l = {}, l.ar = 1, l.pv = 1, l)),
                                    Y: {fa: mb(a, v) || void 0},
                                    G: (m = {}, m["page-url"] = G, m)
                                }, {id: sa.bh, ba: "0"})["catch"](D(a, "r.tim.ng2"))
                            }
                        }
                        d()
                    }
                }
            }, 7);
            va.ci = [wb];
            Cb.push(zu);
            U.push(Au);
            Cb.push(Ys);
            U.push(Fu);
            zb(oh, 8);
            ba("f", oh, 3);
            ba("n", oh, 5);
            Cc.push(function (a) {
                return C("fip", function (c) {
                    if (!$k(c) || Kd(c)) {
                        var b = Ra(c);
                        if (!b.o("fip")) {
                            var d = t(hb(t(function (e, f) {
                                return C("fip." + f, e)(c)
                            }, F(kr, null))), qd("-"))(a);
                            b.C("fip", d)
                        }
                    }
                })
            }([Hu, Gu, function (a) {
                var c = n(a, "ApplePaySession"), b = S(a).protocol;
                a = c && "https:" === b && !eb(a) ? c : void 0;
                c = "";
                if (!a) return c;
                try {
                    c = "" + a.canMakePayments();
                    b = "";
                    var d = a.supportsVersion;
                    if (T(d)) for (var e = 1; 20 >= e; e += 1) b += d.call(a, e) ? "" + e : "0";
                    return b + c
                } catch (f) {
                    return c
                }
            }, function (a) {
                a = n(a, "navigator") || {};
                return a.doNotTrack || a.msDoNotTrack || "unknown"
            }, function (a) {
                if (a = Ws(a)) try {
                    for (var c = [], b = 0; b < al.length; b += 1) {
                        var d = a(al[b]);
                        c.push(d)
                    }
                    var e = c
                } catch (f) {
                    e = []
                } else e = [];
                return e ? J("x", e) :
                    ""
            }, function (a) {
                var c = void 0;
                void 0 === c && (c = Du);
                var b = n(a, "navigator") || {};
                c = A(u(b, n), c);
                c = J("x", c);
                try {
                    var d = n(a, "navigator.getGamepads");
                    var e = Ma(d, "getGamepads") && a.navigator.getGamepads() || []
                } catch (f) {
                    e = []
                }
                return c + "x" + Pa(e)
            }, Bu, function (a) {
                a = n(a, "screen") || {};
                return J("x", A(u(a, n), Cu))
            }, function (a) {
                return J("x", jm(a) || [])
            }, function (a) {
                a = Am(a);
                return ca(a) ? J("x", a) : a
            }, function (a) {
                return (a = Cm(a)) ? t(Cr, Rg(O), hb(E(["", ["matches", "media"]], lm)), u("x", J))(a) : ""
            }]));
            zb(function (a) {
                return {
                    Z: function (c,
                                 b) {
                        var d = c.H, e = Ra(a).o("fip");
                        e && d && (d.C("fip", e), $d(c, "fip", hd(e)));
                        b()
                    }
                }
            }, 9);
            ba("h", function (a) {
                return {
                    Ea: function (c, b) {
                        var d = c.tj;
                        re(c) && d && H(a).C("isEU", n(d, "settings.eu"));
                        b()
                    }
                }
            }, 3);
            Cc.push(cu);
            We.push(Iu);
            Cb.push(Nu);
            U.push(Ou);
            fc({ak: {ia: "yaDisableGDPR"}, bk: {ia: "yaGDPRLang"}});
            Ce.push(function (a, c) {
                return {Z: E([a, c], Yl)}
            });
            mg.push("gdpr", "gdpr_popup");
            lg.push(function (a, c) {
                var b = Zd(a);
                b = lf(b);
                if (Z(uc(Ru), b).length) return !0;
                b = c(a, "gdpr");
                return K(b, [Ec, Qu])
            });
            Ce.push(function (a) {
                return {
                    Z: function (c,
                                 b) {
                        var d = c.ja || {}, e;
                        (e = n(a, "document.referrer")) ? (e = Hc(a, e).host, e = gj(e), e = dl + "." + (e || dt)) : e = bc;
                        c.ja = y(d, {Ci: [e]});
                        b()
                    }
                }
            });
            (function () {
                mg.push("_ym_debug_build");
                Xl("init", function (a, c) {
                    var b = "1" === c.ba, d = eg(c);
                    return eb(a) || b || d || !Su(a) ? !1 : !0
                })
            })();
            zb(Cl, 5);
            ba("1", Cl, 6);
            va.c = Xc;
            Ea.c = Eb;
            ba("h", function (a) {
                return {
                    Z: function (c, b) {
                        re(c) ? xl(a).then(Kl, Nl).then(function (d) {
                            c.G || (c.G = {});
                            c.G.uah = d;
                            b()
                        }, b) : b()
                    }
                }
            }, 7);
            (function () {
                function a(c) {
                    var b = qc("canvas", c.document);
                    if (b && (b = Ic(b))) {
                        var d = Je(c) || Qc(c),
                            e = d[0];
                        d = d[1];
                        if (.3 <= Cj(c, b, {Qa: d, pd: e}) / d * e) return H(c).C("hc", 1), !0
                    }
                    return !1
                }

                U.push(C("hcp", function (c) {
                    a(c) || V(c, u(c, a), 3E3)
                }))
            })();
            U.push(C("p.ot", Uu));
            U.push(function (a, c) {
                var b = zl(a), d = N(c), e = b[d];
                e || (e = {}, b[d] = e);
                e.Ei = !0;
                if (b = e.xg) d = Al(a), z(d, b)
            });
            We.push(function (a) {
                var c = n(a, "ym");
                if (c) {
                    var b = n(c, "a");
                    b || (c.a = [], b = c.a);
                    var d = Al(a);
                    xe(a, b, function (e) {
                        e.Aa.D(d)
                    }, !0)
                }
            });
            if (window.Ya && df) {
                var Dl = sa.uc;
                window.Ya[Dl] = df;
                ls(window);
                z(t(Oc([window, window.Ya[Dl]]), ha), Ug)
            }
            z(t(Oc([window]), ha), We)
        })()
    } catch (df) {
    }
    ;
}).call(this)
//# sourceMappingURL=tag.js.map
