// main script
(function () {
  "use strict";

  AOS.init({
    once: true,
  });

  // Back to Top
  const bctBtn = document.querySelector(".back-to-top");
  if (bctBtn) {
    window.addEventListener("scroll", () => {
      if (document.documentElement.scrollTop <= 100) {
        bctBtn.classList.remove("active");
      } else {
        bctBtn.classList.add("active");
      }
    });

    bctBtn.addEventListener("click", () => {
      document.documentElement.scrollTop = 0;
    });
  }

  // Sidebar in Mobile Devices Hide & Show
  const sidebar = document.querySelector("#sidebarContent");
  const navbarToggler = document.querySelector(".navbar-toggler");
  navbarToggler.addEventListener("click", (e) => {
    e.preventDefault();

    function closeSidebar() {
      navbarToggler.classList.remove("active");
      sidebar.classList.remove("active");

      // Prevent Body Scrolling
      document.body.classList.remove("navbar-show");
      document.body.style.removeProperty("padding-right");

      // Remove Backdrop
      if (document.querySelector(".tf-backdrop")) {
        document.querySelector(".tf-backdrop").remove();
      }
    }

    function showSidebar() {
      navbarToggler.classList.add("active");
      sidebar.classList.add("active");

      // Close Sidebar If user click outside of sidebar element
      const backdrop = document.createElement("div");
      backdrop.setAttribute(
        "style",
        "width: 100vw;height: 100vh;background-color:#000000;opacity: 0.6;position:fixed;z-index:555;left:0;top:0;"
      );
      backdrop.setAttribute("class", "tf-backdrop");
      if (!document.querySelector(".tf-backdrop")) {
        document.body.insertBefore(backdrop, sidebar.children[-1]);
      }
    }

    if (navbarToggler.classList.contains("active")) {
      closeSidebar();
    } else {
      showSidebar();
      document.querySelector(".tf-backdrop").addEventListener("click", () => {
        closeSidebar();
      });
    }

    window.addEventListener("resize", () => {
      if (screen.width >= 1200) {
        closeSidebar();
      }
    });
  });
})();
