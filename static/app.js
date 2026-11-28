(function () {
  // Menu mobile
  var burger = document.querySelector('.burger'), menu = document.getElementById('menu');
  if (burger && menu) {
    burger.addEventListener('click', function () {
      var open = menu.classList.toggle('open');
      burger.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }
  // Isi cepat formulir
  var el = document.getElementById('presets');
  if (el) {
    var presets = JSON.parse(el.textContent);
    document.querySelectorAll('[data-preset]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var p = presets[btn.dataset.preset] || {};
        Object.keys(p).forEach(function (k) { var f = document.getElementById(k); if (f) f.value = p[k]; });
      });
    });
  }
  // Gulir ke hasil
  var hasil = document.getElementById('hasil');
  if (hasil) { hasil.scrollIntoView({ behavior: 'smooth', block: 'start' }); hasil.focus({ preventScroll: true }); }
})();
